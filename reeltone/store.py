from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from .contracts import identifier, option, safe_filename, text
from .errors import Absent, Conflict, Limit


COLLECTION_LIMIT = 48
SCAN_LIMIT = 48
SCAN_BYTES_PER_REEL = 480 * 1024 * 1024
TAKE_LIMIT = 120
PROOF_LIMIT = 100
CUE_LIMIT = 100
RECIPE_HISTORY_LIMIT = 200
PACKET_LIMIT = 30
JOIN_LIMIT = 24
MAX_NAMED_RANGES = 120


SCHEMA = """
CREATE TABLE IF NOT EXISTS reels (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    accession TEXT NOT NULL,
    gauge TEXT NOT NULL,
    note TEXT NOT NULL,
    synthetic INTEGER NOT NULL DEFAULT 0,
    demo_key TEXT UNIQUE,
    created TEXT NOT NULL,
    updated TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS scans (
    id TEXT PRIMARY KEY,
    reel_id TEXT NOT NULL REFERENCES reels(id),
    title TEXT NOT NULL,
    source_note TEXT NOT NULL,
    raw BLOB NOT NULL,
    metadata TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    origin_kind TEXT NOT NULL DEFAULT 'imported',
    created TEXT NOT NULL,
    UNIQUE(reel_id, sha256)
);
CREATE TABLE IF NOT EXISTS recipes (
    id TEXT PRIMARY KEY,
    scan_id TEXT NOT NULL REFERENCES scans(id),
    seq INTEGER NOT NULL,
    settings TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    label TEXT NOT NULL,
    theme TEXT NOT NULL,
    created TEXT NOT NULL,
    UNIQUE(scan_id, seq)
);
CREATE TABLE IF NOT EXISTS takes (
    id TEXT PRIMARY KEY,
    scan_id TEXT NOT NULL REFERENCES scans(id),
    recipe_id TEXT NOT NULL REFERENCES recipes(id),
    title TEXT NOT NULL,
    settings TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft',
    wav BLOB NOT NULL,
    wav_sha256 TEXT NOT NULL,
    created TEXT NOT NULL,
    updated TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS proofs (
    id TEXT PRIMARY KEY,
    scan_id TEXT NOT NULL REFERENCES scans(id),
    recipe_id TEXT NOT NULL REFERENCES recipes(id),
    title TEXT NOT NULL,
    category TEXT NOT NULL,
    note TEXT NOT NULL,
    evidence TEXT NOT NULL,
    primary_file BLOB NOT NULL,
    primary_sha256 TEXT NOT NULL,
    secondary_file BLOB,
    secondary_sha256 TEXT,
    created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cues (
    id TEXT PRIMARY KEY,
    reel_id TEXT NOT NULL REFERENCES reels(id),
    scan_id TEXT NOT NULL REFERENCES scans(id),
    title TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    y0 INTEGER NOT NULL,
    y1 INTEGER NOT NULL,
    anchor TEXT,
    note TEXT NOT NULL,
    created TEXT NOT NULL,
    updated TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS packets (
    id TEXT PRIMARY KEY,
    reel_id TEXT NOT NULL REFERENCES reels(id),
    title TEXT NOT NULL,
    filename TEXT NOT NULL,
    summary TEXT NOT NULL,
    zip_data BLOB NOT NULL,
    sha256 TEXT NOT NULL,
    created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS markers (
    id TEXT PRIMARY KEY,
    scan_id TEXT NOT NULL REFERENCES scans(id),
    title TEXT NOT NULL,
    kind TEXT NOT NULL,
    display TEXT NOT NULL,
    notes TEXT NOT NULL,
    created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS joins (
    id TEXT PRIMARY KEY,
    reel_id TEXT NOT NULL REFERENCES reels(id),
    output_scan_id TEXT NOT NULL REFERENCES scans(id),
    head_scan_id TEXT NOT NULL REFERENCES scans(id),
    tail_scan_id TEXT NOT NULL REFERENCES scans(id),
    title TEXT NOT NULL,
    manifest TEXT NOT NULL,
    created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reel_id TEXT NOT NULL REFERENCES reels(id),
    scan_id TEXT REFERENCES scans(id),
    subject_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    detail TEXT NOT NULL,
    time TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS scans_by_reel ON scans(reel_id, created);
CREATE INDEX IF NOT EXISTS recipes_by_scan ON recipes(scan_id, seq DESC);
CREATE INDEX IF NOT EXISTS takes_by_scan ON takes(scan_id, created DESC);
CREATE INDEX IF NOT EXISTS proofs_by_scan ON proofs(scan_id, created DESC);
CREATE INDEX IF NOT EXISTS cues_by_reel ON cues(reel_id, scan_id, y0, y1);
CREATE INDEX IF NOT EXISTS packets_by_reel ON packets(reel_id, created DESC);
CREATE INDEX IF NOT EXISTS markers_by_scan ON markers(scan_id, created DESC);
CREATE INDEX IF NOT EXISTS joins_by_reel ON joins(reel_id, created DESC);
CREATE INDEX IF NOT EXISTS events_by_reel ON events(reel_id, id DESC);
"""

MIGRATIONS = (
    # 1.0 libraries predate the origin_kind column; default keeps old scans imports.
    "ALTER TABLE scans ADD COLUMN origin_kind TEXT NOT NULL DEFAULT 'imported'",
)

REEL_FIELDS = "id,title,accession,gauge,note,synthetic,demo_key,created,updated"
SCAN_FIELDS = "id,reel_id,title,source_note,metadata,sha256,origin_kind,created"
RECIPE_FIELDS = "id,scan_id,seq,settings,fingerprint,label,theme,created"
TAKE_FIELDS = "id,scan_id,recipe_id,title,settings,payload,status,wav_sha256,created,updated"
PROOF_FIELDS = "id,scan_id,recipe_id,title,category,note,evidence,primary_sha256,secondary_sha256,created"
CUE_FIELDS = "id,reel_id,scan_id,title,kind,status,y0,y1,anchor,note,created,updated"
PACKET_FIELDS = "id,reel_id,title,filename,summary,sha256,created"
MARKER_FIELDS = "id,scan_id,title,kind,display,notes,created"
JOIN_FIELDS = "id,reel_id,output_scan_id,head_scan_id,tail_scan_id,title,manifest,created"

GAUGES = ("16 mm", "35 mm", "Other")
RECIPE_THEMES = ("intake", "manual", "restored", "reference")
TAKE_STATES = ("draft", "reviewed", "hold")
PROOF_CATEGORIES = ("image", "exposure", "blemish", "comparison", "scope", "perforations", "sample")
CUE_KINDS = ("excerpt", "splice", "damage", "silence", "sync", "leader")
CUE_STATES = ("open", "checked", "omit")
MARKER_KINDS = ("reference", "track", "splice", "surface")


def utc_text():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def key():
    return uuid4().hex


def serialize(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False, sort_keys=True)


def deserialize(row, fields=(), *, bools=()):
    doc = dict(row)
    for field in fields:
        if doc[field] is not None:
            doc[field] = json.loads(doc[field])
    for field in bools:
        doc[field] = bool(doc[field])
    return doc


def one(connection, sql, parameters, label):
    row = connection.execute(sql, parameters).fetchone()
    if row is None:
        raise Absent(f"找不到{label}。")
    return row


def bounded_count(connection, table, parent_column, value, maximum, message):
    count = connection.execute(f"SELECT count(*) FROM {table} WHERE {parent_column}=?", (value,)).fetchone()[0]
    if count >= maximum:
        raise Limit(message)


def event(connection, reel, scan, subject, kind, detail):
    now = utc_text()
    connection.execute("INSERT INTO events(reel_id,scan_id,subject_id,kind,detail,time) VALUES(?,?,?,?,?,?)",
                       (reel, scan, subject, kind, serialize(detail), now))
    connection.execute("UPDATE reels SET updated=? WHERE id=?", (now, reel))


class Catalog:
    def __init__(self, directory):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "library.sqlite3"
        with self.database() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(SCHEMA)
            self._migrate(connection)

    def _migrate(self, connection):
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(scans)")}
        if "origin_kind" not in columns:
            connection.execute(MIGRATIONS[0])

    @contextmanager
    def database(self, write=False):
        connection = sqlite3.connect(self.path, timeout=8, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            if write:
                connection.execute("BEGIN IMMEDIATE")
            yield connection
            if write:
                connection.commit()
        except BaseException:
            if write and connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _summary_reel(connection, reel):
        row = one(connection, f"SELECT {REEL_FIELDS} FROM reels WHERE id=?", (reel,), "档案卷")
        return deserialize(row, bools=("synthetic",))

    def get_reel(self, reel):
        with self.database() as connection:
            return self._summary_reel(connection, identifier(reel))

    def reels(self):
        query = f"""SELECT {REEL_FIELDS},
             (SELECT count(*) FROM scans WHERE reel_id=reels.id) scan_count,
             (SELECT count(*) FROM cues WHERE reel_id=reels.id AND status='open') open_cues,
             (SELECT count(*) FROM packets WHERE reel_id=reels.id) packet_count
             FROM reels ORDER BY updated DESC, id"""
        with self.database() as connection:
            return [deserialize(row, bools=("synthetic",)) for row in connection.execute(query)]

    def demo_reel(self, demo_key):
        with self.database() as connection:
            row = connection.execute("SELECT id FROM reels WHERE demo_key=?", (demo_key,)).fetchone()
            return row["id"] if row else None

    def _insert_reel(self, connection, title, accession, gauge, note, synthetic, demo_key):
        previous = None
        if demo_key:
            previous = connection.execute("SELECT id FROM reels WHERE demo_key=?", (demo_key,)).fetchone()
        if previous:
            return previous["id"], False
        if connection.execute("SELECT count(*) FROM reels").fetchone()[0] >= COLLECTION_LIMIT:
            raise Limit(f"本工作库最多保留 {COLLECTION_LIMIT} 卷。")
        now, reel_id = utc_text(), key()
        connection.execute(f"INSERT INTO reels({REEL_FIELDS}) VALUES(?,?,?,?,?,?,?,?,?)",
                           (reel_id, title, accession, gauge, note, synthetic, demo_key, now, now))
        event(connection, reel_id, None, reel_id, "reel.created", {"title": title, "synthetic": synthetic})
        return reel_id, True

    def create_reel(self, values):
        title = text(values.get("title", ""), "档案卷标题")
        accession = text(values.get("accession", ""), "入藏编号", maximum=70, required=False)
        gauge = option(values.get("gauge", "16 mm"), GAUGES, "胶片规格")
        note = text(values.get("note", ""), "档案备注", maximum=800, required=False, multiline=True)
        with self.database(write=True) as connection:
            reel_id, _ = self._insert_reel(connection, title, accession, gauge, note, False, None)
            return self._summary_reel(connection, reel_id)

    def seed_reel(self, values, drafts):
        with self.database(write=True) as connection:
            reel_id, created = self._insert_reel(connection, values["title"], values["accession"], values["gauge"],
                                                 values["note"], True, values["demo_key"])
            if created:
                for draft in drafts:
                    self._insert_scan(connection, reel_id, **draft)
            return self._summary_reel(connection, reel_id), created

    @staticmethod
    def _get_scan(connection, scan_id, raw=False):
        extras = ",raw" if raw else ""
        row = one(connection, f"SELECT {SCAN_FIELDS}{extras} FROM scans WHERE id=?", (identifier(scan_id),), "扫描")
        return deserialize(row, ("metadata",))

    def get_scan(self, scan_id, *, raw=False):
        with self.database() as connection:
            return self._get_scan(connection, scan_id, raw)

    def list_scans(self, reel):
        with self.database() as connection:
            self._summary_reel(connection, identifier(reel))
            rows = connection.execute(f"SELECT {SCAN_FIELDS} FROM scans WHERE reel_id=? ORDER BY created,id", (reel,))
            return [deserialize(row, ("metadata",)) for row in rows]

    def _insert_scan(self, connection, reel, *, raster, title, note, recipe, origin_kind="imported"):
        previous = connection.execute("SELECT id FROM scans WHERE reel_id=? AND sha256=?", (reel, raster.digest)).fetchone()
        if previous:
            return previous["id"], False
        bounded_count(connection, "scans", "reel_id", reel, SCAN_LIMIT, f"每卷最多收录 {SCAN_LIMIT} 个扫描。")
        old_bytes = connection.execute("SELECT coalesce(sum(length(raw)),0) FROM scans WHERE reel_id=?", (reel,)).fetchone()[0]
        if old_bytes + len(raster.raw) > SCAN_BYTES_PER_REEL:
            raise Limit("每卷的源扫描总量最多为 480 MiB。")
        scan_id = key()
        connection.execute(f"INSERT INTO scans({SCAN_FIELDS},raw) VALUES(?,?,?,?,?,?,?,?,?)",
                           (scan_id, reel, title, note, serialize(raster.metadata()), raster.digest,
                            origin_kind, utc_text(), raster.raw))
        event(connection, reel, scan_id, scan_id,
              "scan.imported" if origin_kind == "imported" else "scan.derived",
              {"title": title, "sha256": raster.digest, "origin": origin_kind})
        self._save_recipe(connection, scan_id, recipe, "入库设置", "intake", 0)
        return scan_id, True

    def add_scan(self, reel, raster, title, note, recipe):
        with self.database(write=True) as connection:
            self._summary_reel(connection, identifier(reel))
            scan_id, created = self._insert_scan(connection, reel, raster=raster, title=title, note=note, recipe=recipe)
            return self._get_scan(connection, scan_id), created

    def add_scan_batch(self, reel, drafts, batch_digest):
        with self.database(write=True) as connection:
            self._summary_reel(connection, identifier(reel))
            receipt = []
            for draft in drafts:
                scan_id, created = self._insert_scan(connection, reel, **draft)
                receipt.append({"scan": self._get_scan(connection, scan_id), "created": created})
            imported_ids = [entry["scan"]["id"] for entry in receipt if entry["created"]]
            if imported_ids:
                event(connection, reel, None, reel, "batch.imported", {"sha256": batch_digest, "created": imported_ids,
                                                                       "duplicate_count": len(receipt) - len(imported_ids)})
            return receipt

    @staticmethod
    def _get_recipe(connection, scan, recipe_id=None):
        if recipe_id is None:
            query = f"SELECT {RECIPE_FIELDS} FROM recipes WHERE scan_id=? ORDER BY seq DESC LIMIT 1"
            params = (identifier(scan),)
        else:
            query = f"SELECT {RECIPE_FIELDS} FROM recipes WHERE scan_id=? AND id=?"
            params = (identifier(scan), identifier(recipe_id))
        return deserialize(one(connection, query, params, "读片配方"), ("settings",))

    def get_recipe(self, scan, recipe_id=None):
        with self.database() as connection:
            return self._get_recipe(connection, scan, recipe_id)

    def recipe_history(self, scan):
        with self.database() as connection:
            self._get_scan(connection, scan)
            rows = connection.execute(f"SELECT {RECIPE_FIELDS} FROM recipes WHERE scan_id=? ORDER BY seq DESC", (scan,))
            return [deserialize(row, ("settings",)) for row in rows]

    def _save_recipe(self, connection, scan_id, recipe, label, theme, expected):
        scan = self._get_scan(connection, scan_id)
        newest = connection.execute("SELECT seq,fingerprint,id FROM recipes WHERE scan_id=? ORDER BY seq DESC LIMIT 1",
                                    (scan_id,)).fetchone()
        sequence = newest["seq"] if newest else 0
        if expected is not None and expected != sequence:
            raise Conflict("该扫描的读片配方已被更新。请重新载入再保存。")
        fingerprint = recipe.fingerprint()
        if newest and newest["fingerprint"] == fingerprint and theme != "restored":
            return self._get_recipe(connection, scan_id, newest["id"]), False
        if sequence >= RECIPE_HISTORY_LIMIT:
            raise Limit(f"每个扫描最多保留 {RECIPE_HISTORY_LIMIT} 个读片版本。")
        recipe_id = key()
        connection.execute(f"INSERT INTO recipes({RECIPE_FIELDS}) VALUES(?,?,?,?,?,?,?,?)",
                           (recipe_id, scan_id, sequence + 1, serialize(recipe.as_dict()), fingerprint, label, theme, utc_text()))
        event(connection, scan["reel_id"], scan_id, recipe_id, "recipe.saved", {"seq": sequence + 1, "label": label})
        return self._get_recipe(connection, scan_id, recipe_id), True

    def save_recipe(self, scan_id, recipe, label, theme, expected):
        option(theme, RECIPE_THEMES, "配方类别")
        with self.database(write=True) as connection:
            return self._save_recipe(connection, scan_id, recipe, label, theme, expected)

    @staticmethod
    def _get_take(connection, take_id, raw=False):
        extras = ",wav" if raw else ""
        row = one(connection, f"SELECT {TAKE_FIELDS}{extras} FROM takes WHERE id=?", (identifier(take_id),), "监听快照")
        return deserialize(row, ("settings", "payload"))

    def get_take(self, take_id, raw=False):
        with self.database() as connection:
            return self._get_take(connection, take_id, raw)

    def list_takes(self, scan_id):
        with self.database() as connection:
            self._get_scan(connection, scan_id)
            rows = connection.execute(f"SELECT {TAKE_FIELDS} FROM takes WHERE scan_id=? ORDER BY created DESC,id", (scan_id,))
            return [deserialize(row, ("settings", "payload")) for row in rows]

    def save_take(self, scan_id, recipe_id, title, settings, payload, wav):
        take = key()
        now = utc_text()
        with self.database(write=True) as connection:
            scan = self._get_scan(connection, scan_id)
            self._get_recipe(connection, scan_id, recipe_id)
            bounded_count(connection, "takes", "scan_id", scan_id, TAKE_LIMIT, f"每份扫描最多保存 {TAKE_LIMIT} 次监听。")
            connection.execute(f"INSERT INTO takes({TAKE_FIELDS},wav) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                               (take, scan_id, recipe_id, title, serialize(settings), serialize(payload), "draft",
                                sha256(wav).hexdigest(), now, now, wav))
            event(connection, scan["reel_id"], scan_id, take, "take.retained", {"title": title, "recipe": recipe_id})
            return self._get_take(connection, take)

    def set_take_status(self, take_id, status):
        status = option(status, TAKE_STATES, "监听状态")
        with self.database(write=True) as connection:
            take = self._get_take(connection, take_id)
            scan = self._get_scan(connection, take["scan_id"])
            if status != take["status"]:
                connection.execute("UPDATE takes SET status=?, updated=? WHERE id=?", (status, utc_text(), take_id))
                event(connection, scan["reel_id"], scan["id"], take_id, "take.status", {"from": take["status"], "to": status})
            return self._get_take(connection, take_id)

    @staticmethod
    def _get_proof(connection, proof_id, raw=False):
        extras = ",primary_file,secondary_file" if raw else ""
        row = one(connection, f"SELECT {PROOF_FIELDS}{extras} FROM proofs WHERE id=?", (identifier(proof_id),), "图像证据")
        return deserialize(row, ("evidence",))

    def get_proof(self, proof_id, raw=False):
        with self.database() as connection:
            return self._get_proof(connection, proof_id, raw)

    def list_proofs(self, scan_id):
        with self.database() as connection:
            self._get_scan(connection, scan_id)
            rows = connection.execute(f"SELECT {PROOF_FIELDS} FROM proofs WHERE scan_id=? ORDER BY created DESC,id", (scan_id,))
            return [deserialize(row, ("evidence",)) for row in rows]

    def save_proof(self, scan_id, recipe_id, title, category, note, evidence, first, second=None):
        option(category, PROOF_CATEGORIES, "证据分类")
        proof_id = key()
        with self.database(write=True) as connection:
            scan = self._get_scan(connection, scan_id)
            self._get_recipe(connection, scan_id, recipe_id)
            bounded_count(connection, "proofs", "scan_id", scan_id, PROOF_LIMIT, f"每份扫描最多 {PROOF_LIMIT} 项图像证据。")
            connection.execute(f"""INSERT INTO proofs({PROOF_FIELDS},primary_file,secondary_file)
                                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                               (proof_id, scan_id, recipe_id, title, category, note, serialize(evidence),
                                sha256(first).hexdigest(), sha256(second).hexdigest() if second else None, utc_text(), first, second))
            event(connection, scan["reel_id"], scan_id, proof_id, "proof.retained", {"title": title, "category": category})
            return self._get_proof(connection, proof_id)

    @staticmethod
    def _get_cue(connection, cue_id):
        row = one(connection, f"SELECT {CUE_FIELDS} FROM cues WHERE id=?", (identifier(cue_id),), "审听标记")
        return deserialize(row, ("anchor",))

    def get_cue(self, cue_id):
        with self.database() as connection:
            return self._get_cue(connection, cue_id)

    def list_cues(self, reel, scan=None):
        with self.database() as connection:
            self._summary_reel(connection, identifier(reel))
            sql = f"SELECT {CUE_FIELDS} FROM cues WHERE reel_id=?"
            query = (reel,)
            if scan is not None:
                sql += " AND scan_id=?"
                query += (identifier(scan),)
            sql += " ORDER BY scan_id,y0,y1,created"
            return [deserialize(row, ("anchor",)) for row in connection.execute(sql, query)]

    def write_cue(self, scan_id, fields, cue_id=None):
        with self.database(write=True) as connection:
            scan = self._get_scan(connection, scan_id)
            reel = scan["reel_id"]
            now = utc_text()
            values = (fields["title"], fields["kind"], fields["status"], fields["y0"], fields["y1"],
                      serialize(fields["anchor"]) if fields["anchor"] else None, fields["note"])
            if cue_id is not None:
                prev = self._get_cue(connection, cue_id)
                if prev["scan_id"] != scan_id:
                    raise Conflict("审听标记属于另一个扫描。")
                if fields.get("previous_updated") != prev["updated"]:
                    raise Conflict("审听标记已变化，请重新载入后再编辑。")
                connection.execute("UPDATE cues SET title=?,kind=?,status=?,y0=?,y1=?,anchor=?,note=?,updated=? WHERE id=?",
                                   (*values, now, cue_id))
            else:
                bounded_count(connection, "cues", "scan_id", scan_id, CUE_LIMIT, f"每份扫描最多保存 {CUE_LIMIT} 个审听标记。")
                cue_id = key()
                connection.execute(f"INSERT INTO cues({CUE_FIELDS}) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                                   (cue_id, reel, scan_id, *values, now, now))
            event(connection, reel, scan_id, cue_id, "cue.saved", {"title": fields["title"], "kind": fields["kind"]})
            return self._get_cue(connection, cue_id)

    def add_marker(self, scan_id, title, kind, display, notes):
        option(kind, MARKER_KINDS, "定位框分类")
        marker = key()
        with self.database(write=True) as connection:
            scan = self._get_scan(connection, scan_id)
            bounded_count(connection, "markers", "scan_id", scan_id, MAX_NAMED_RANGES, "该扫描的定位框已达上限。")
            connection.execute(f"INSERT INTO markers({MARKER_FIELDS}) VALUES(?,?,?,?,?,?,?)",
                               (marker, scan_id, title, kind, serialize(display), notes, utc_text()))
            event(connection, scan["reel_id"], scan_id, marker, "region.saved", {"title": title, "kind": kind})
            return deserialize(one(connection, f"SELECT {MARKER_FIELDS} FROM markers WHERE id=?", (marker,), "定位框"), ("display",))

    def list_markers(self, scan_id):
        with self.database() as connection:
            self._get_scan(connection, scan_id)
            rows = connection.execute(f"SELECT {MARKER_FIELDS} FROM markers WHERE scan_id=? ORDER BY created DESC,id", (scan_id,))
            return [deserialize(row, ("display",)) for row in rows]

    @staticmethod
    def _get_packet(connection, packet_id, raw=False):
        extras = ",zip_data" if raw else ""
        row = one(connection, f"SELECT {PACKET_FIELDS}{extras} FROM packets WHERE id=?", (identifier(packet_id),), "审听交付包")
        return deserialize(row, ("summary",))

    def get_packet(self, packet_id, raw=False):
        with self.database() as connection:
            return self._get_packet(connection, packet_id, raw)

    def list_packets(self, reel_id):
        with self.database() as connection:
            self._summary_reel(connection, identifier(reel_id))
            rows = connection.execute(f"SELECT {PACKET_FIELDS} FROM packets WHERE reel_id=? ORDER BY created DESC,id", (reel_id,))
            return [deserialize(row, ("summary",)) for row in rows]

    def save_packet(self, reel_id, title, filename, summary, archive, *, take_revisions, cue_revisions):
        packet = key()
        with self.database(write=True) as connection:
            self._summary_reel(connection, identifier(reel_id))
            bounded_count(connection, "packets", "reel_id", reel_id, PACKET_LIMIT, "该卷的审听交付包已达上限。")
            for take_id, updated in take_revisions.items():
                existing = one(connection, "SELECT updated,status FROM takes WHERE id=?", (take_id,), "监听快照")
                if updated != existing["updated"] or existing["status"] != "reviewed":
                    raise Conflict("监听复核状态在打包期间变化，请重新选择交付内容。")
            for cue_id, updated in cue_revisions.items():
                existing = one(connection, "SELECT updated FROM cues WHERE id=?", (cue_id,), "审听标记")
                if updated != existing["updated"]:
                    raise Conflict("审听标记在打包期间变化，请重新检查交付片段。")
            connection.execute(f"INSERT INTO packets({PACKET_FIELDS},zip_data) VALUES(?,?,?,?,?,?,?,?)",
                               (packet, reel_id, title, safe_filename(filename, "review.zip"), serialize(summary),
                                sha256(archive).hexdigest(), utc_text(), archive))
            event(connection, reel_id, None, packet, "packet.retained", {"title": title, "asset_count": summary["asset_count"]})
            return self._get_packet(connection, packet)

    def save_join(self, reel_id, *, raster, title, note, recipe, head_id, tail_id, manifest):
        join_id = key()
        with self.database(write=True) as connection:
            self._summary_reel(connection, identifier(reel_id))
            head = self._get_scan(connection, head_id)
            tail = self._get_scan(connection, tail_id)
            if head["reel_id"] != reel_id or tail["reel_id"] != reel_id:
                raise Conflict("接续的两张扫描须属于同一个档案卷。")
            bounded_count(connection, "joins", "reel_id", reel_id, JOIN_LIMIT, f"每卷最多保留 {JOIN_LIMIT} 次接续。")
            output_id, created = self._insert_scan(connection, reel_id, raster=raster, title=title, note=note,
                                                   recipe=recipe, origin_kind="derived_join")
            if not created:
                raise Conflict("接续结果与卷内已有图像字节相同，已保留原条目，未重复保存。")
            connection.execute(f"INSERT INTO joins({JOIN_FIELDS}) VALUES(?,?,?,?,?,?,?,?)",
                               (join_id, reel_id, output_id, head_id, tail_id, title, serialize(manifest), utc_text()))
            event(connection, reel_id, output_id, join_id, "join.retained",
                  {"title": title, "head": head_id, "tail": tail_id, "segments": len(manifest["segments"])})
            return self._get_join(connection, join_id)

    def _get_join(self, connection, join_id):
        row = one(connection, f"SELECT {JOIN_FIELDS} FROM joins WHERE id=?", (identifier(join_id),), "接续记录")
        return deserialize(row, ("manifest",))

    def get_join(self, join_id):
        with self.database() as connection:
            return self._get_join(connection, join_id)

    def join_for_scan(self, scan_id):
        with self.database() as connection:
            self._get_scan(connection, scan_id)
            row = connection.execute("SELECT id FROM joins WHERE output_scan_id=?", (identifier(scan_id),)).fetchone()
            return self._get_join(connection, row["id"]) if row else None

    def list_joins(self, reel_id):
        with self.database() as connection:
            self._summary_reel(connection, identifier(reel_id))
            rows = connection.execute(f"SELECT {JOIN_FIELDS} FROM joins WHERE reel_id=? ORDER BY created,id", (reel_id,))
            return [deserialize(row, ("manifest",)) for row in rows]

    def audit(self, reel_id, before=None, limit=30):
        query = "SELECT id,reel_id,scan_id,subject_id,kind,detail,time FROM events WHERE reel_id=?"
        params = (identifier(reel_id),)
        if before is not None:
            query += " AND id < ?"
            params += (before,)
        with self.database() as connection:
            self._summary_reel(connection, reel_id)
            rows = connection.execute(query + " ORDER BY id DESC LIMIT ?", (*params, limit))
            return [deserialize(row, ("detail",)) for row in rows]
