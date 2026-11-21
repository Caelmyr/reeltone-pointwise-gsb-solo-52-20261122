import {action, check, esc, field, fmt, form, image, jsonDetails, link, listTable, metrics, percent, record, reelURL, REVIEW, select, span, submit, TAKE, CUE, CUE_STATUS, textarea} from './ui.js';
import {blank, currentBox} from './film.js';
import {rowsFields} from './panels.js';

export function cueForm(state, existing = null) {
    const scan = state.scan.scan;
    const source = existing ? [existing.y0,existing.y1] : currentBox(state).filter((_,n)=>n%2===1);
    const fields = field('title','标记名称',existing?.title || '',{full:true,maxlength:90})+
        rowsFields(source,scan.metadata.height)+select('kind','标记类型',Object.entries(CUE),existing?.kind || 'excerpt')+
        select('status','标记状态',Object.entries(CUE_STATUS),existing?.status || 'open')+
        select('take','关联监听（选填）',[['','无'],...state.takes.map(t=>[t.id,t.title])],existing?.anchor?.take || '',true)+
        textarea('note','证据说明',existing?.note || '',600);
    return form('cue-form',fields,submit(existing?'更新审听标记':'保存审听标记'));
}

function rowCoverage(data, height) {
    const marks = [...data.unheard_spans.map(rows=>[...rows,'unheard']),...data.alternate_spans.map(item=>[...item.rows,'multiple'])];
    marks.sort(([a],[b])=>a-b);
    const cells = [];
    let cursor = 0;
    for (const [a,b,label] of marks) {
        if (cursor < a) cells.push([cursor,a,'single']);
        cells.push([a,b,label]);cursor=b;
    }
    if (cursor<height) cells.push([cursor,height,'single']);
    const label = {single:'已复核',unheard:'未覆盖',multiple:'多个已复核版本'};
    return `<div class="timeline-track" aria-label="覆盖条">${cells.map(([a,b,kind])=>`<span class="${kind}" style="width:${100*(b-a)/height}%" title="${label[kind]} · ${fmt(a,0)}–${fmt(b,0)}"></span>`).join('')}</div>`;
}

function readinessSummary(state) {
    const report = state.desk.inventory;
    if (!report) return '<div class="slot-empty">正在计算审听覆盖…</div>';
    const reel = state.reel.reel.id;
    const controls = [link(reelURL(reel,'/readiness/html'),'打印审听覆盖',{target:true}),link(reelURL(reel,'/readiness/csv'),'下载标记检查 CSV'),
        action('view-readiness','打开覆盖图')].join('');
    const sections = report.runs.map(run => {
        const warnings = run.multiple_read_clocks ? ' · 含多个读片时钟' : '';
        return `<div class="record"><div style="flex:1"><strong>${esc(run.scan.title)}</strong><small>${percent(run.coverage.reviewed_fraction)} 行已复核 · ${run.cue_ready_count} 项可交付${warnings}</small>${rowCoverage(run.coverage,run.scan.height)}</div><div class="record-tools">${action('open-native','查看原片',{id:run.scan.id,row:0})}</div></div>`;
    });
    return `<section class="box full"><h2>审听覆盖</h2>${metrics([['覆盖行数',`${fmt(report.reviewed_rows,0)} / ${fmt(report.all_native_rows,0)}`],['已保存监听',report.retained_takes],['待核标记',report.pending_cues],['可交付标记',report.ready_cues]])}<div class="data-list">${sections.join('') || '<p class="panel-note">尚未导入扫描。</p>'}</div><div class="form-footer">${controls}</div></section>`;
}

function cueRows(state) {
    const cues = state.desk.cues || [];
    const reports = (state.desk.inventory?.runs || []).flatMap(row=>row.cues);
    const review = new Map(reports.map(entry=>[entry.id,entry]));
    const body = cues.map(cue=>{
        const origin = state.reel.scans.find(scan=>scan.id===cue.scan_id);
        const progress = review.get(cue.id)?.outcome;
        const subtitle = `${origin?.title || ''} · ${CUE[cue.kind]} · ${span([cue.y0,cue.y1])} 行 · ${progress ? REVIEW[progress] : CUE_STATUS[cue.status]}`;
        return record(cue.title,subtitle,`${action('cue-view','定位',{id:cue.id})}${action('edit-cue','编辑',{id:cue.id})}`);
    }).join('');
    const create = state.scan ? action('new-cue','为当前扫描添加标记',{},'strong') : '';
    return `<section class="box"><h2>审听标记</h2><div class="form-footer">${create}</div><div class="data-list">${body || '<p class="panel-note">暂无原片行标记。</p>'}</div></section>`;
}

function deliverableTitle(take,state) {
    const lookup = state.reel.scans.find(scan=>scan.id===take.scan_id);
    return `${take.title} · ${lookup?.title || ''}`;
}

function packBuilder(state) {
    const ready = state.desk.takes.filter(take=>take.status==='reviewed');
    const timeline = state.desk.items.map((item,index)=>{
        const selected = ready.find(take=>take.id===item.take);
        const available = state.desk.cues.filter(cue=>cue.scan_id===selected?.scan_id && cue.status!=='omit' && selected?.payload.audio.source_rows[0]<=cue.y0 && selected?.payload.audio.source_rows[1]>=cue.y1);
        return `<div class="playlist-item" data-item="${index}">${select('playtake','已复核监听',[['','请选择'],...ready.map(take=>[take.id,deliverableTitle(take,state)])],selected?.id || '')}${select('playcue','裁切范围',[['','完整监听'],...available.map(cue=>[cue.id,cue.title])],item.cue || '')}${field('playgap','前置留白 ms',item.gap_ms,{type:'number',min:0,max:2000,step:1})}<div class="row-tools">${action('item-up','上移',{n:index})}${action('item-down','下移',{n:index})}${action('item-remove','移除',{n:index})}</div></div>`;
    }).join('');
    const proofs = state.desk.proofs.map(proof=>{
        const origin = state.reel.scans.find(scan=>scan.id===proof.scan_id);
        return `<label class="check"><input type="checkbox" name="proof" value="${proof.id}" ${state.desk.checkedProofs.includes(proof.id)?'checked':''}> ${esc(proof.title)} · ${esc(origin?.title || '')}</label>`;
    }).join('');
    const header = `<div class="fields">${field('title','交付包名称',state.desk.title,{full:true,maxlength:100})}${check('continuous','附串听 WAV',state.desk.continuous)}${check('include_cues','附审听标记表',state.desk.include_cues)}${check('include_sources','附原始扫描',state.desk.include_sources,true)}${select('rate','串听采样率',[[24000,'24,000 Hz'],[48000,'48,000 Hz'],[16000,'16,000 Hz'],[44100,'44,100 Hz']],state.desk.rate,true)}</div>`;
    const items = `<h3 style="margin-top:12px">监听顺序</h3><div class="playqueue">${timeline || '<p class="panel-note">可加入已复核的监听，也可以只交付图像证据和标记。</p>'}</div>${action('item-add','添加监听顺序项')}`;
    const evidence = `<h3 style="margin-top:12px">图像证据</h3><div class="checklist">${proofs || '<p class="panel-note">还没有保存的图像证据。</p>'}</div>`;
    return `<section class="box full"><h2>制作审听交付包</h2><form id="packet-form">${header}${items}${evidence}<div class="form-footer">${submit('保存审听交付包')}</div></form></section>`;
}

function galleryPackets(state) {
    const list = state.reel.packets.filter(packet=>packet.summary.schema==='reeltone-gallery-v1').map(packet=>
        record(packet.title,`${packet.summary.span_count} 个遮盖区间 · ${fmt(packet.summary.held_rows,0)} 行硬遮盖 · ${fmt(packet.summary.bytes/1024)} KiB`,
            `${action('verify-retained','校验',{id:packet.id})}${link(`/api/packets/${packet.id}/download`,'下载阅览 ZIP')}`)).join('');
    return `<section class="box"><h2>展厅阅览遮盖包</h2><p class="panel-note">包内只有处理后的边带图、遮盖监听和原片行区间说明；不含原始扫描。遮盖区无法从包内还原。</p><div class="data-list">${list || '<p class="panel-note">还未制作阅览遮盖包。在读片台的「阅览遮盖」页签按原片行圈选并导出。</p>'}</div></section>`;
}

function deliverables(state) {
    const list = state.reel.packets.filter(packet=>packet.summary.schema!=='reeltone-gallery-v1').map(packet=>record(packet.title,`${packet.summary.clip_count} 个监听片段 · ${packet.summary.images} 幅图像证据 · ${fmt(packet.summary.bytes/1024)} KiB`,
        `${action('verify-retained','校验',{id:packet.id})}${link(`/api/packets/${packet.id}/download`,'下载 ZIP')}`)).join('');
    return `<section class="box"><h2>已交付的审听包</h2><div class="form-footer">${action('verify-zip','核验收到的 ZIP')}${action('events','档案变更记录')}</div><div class="data-list">${list || '<p class="panel-note">还未制作审听交付包。</p>'}</div></section>`;
}

export function deskPage(state) {
    if (!state.reel) return blank(state);
    const reel = state.reel.reel;
    if (!state.desk) return '<div class="loading">正在汇总档案卷记录…</div>';
    return `<header class="scene-header"><div><h1>审听与交付</h1><div class="scene-sub"><span>${esc(reel.title)}</span><span>${esc(reel.accession || '未填入藏号')}</span></div></div><div class="scene-actions">${action('workspace','回到读片台',{view:'scan'})}</div></header><div class="desk-grid">${readinessSummary(state)}${cueRows(state)}${deliverables(state)}${galleryPackets(state)}${packBuilder(state)}</div>`;
}

export function galleryReady(packet, manifest) {
    const rows = manifest.redaction.spans.map(item=>[item.label,span(item.rows),item.note || '—']);
    return metrics([['遮盖区间',manifest.redaction.spans.length],['硬遮盖行',manifest.redaction.hard_held_rows],
        ['衔接行数',manifest.redaction.feather_rows],['监听帧数',fmt(manifest.audio_frames,0)]])+
        `<p class="panel-note status-good">遮盖区内图像为基准重建、监听样本严格为零；总时长与原片一致，行号可对照行区间表核对。</p>
        <div class="form-footer">${link(`/api/packets/${packet.id}/download`,'下载阅览 ZIP',{css:'strong'})}${action('verify-retained','校验内容',{id:packet.id})}</div>`+
        (rows.length ? listTable(['区间','原片行','说明'],rows) : '');
}

export function zipSummary(result) {
    const files = result.files.map(file=>[file.path,file.status,fmt(file.bytes,0)]);
    const state = result.verified && result.receipt_matches !== false ? '通过校验' : '发现内容差异';
    const notes = result.relationships.length ? jsonDetails(result.relationships,'来源关系异常') : '';
    return `<p class="${result.verified?'status-good':'status-error'}">${state} · ${esc(result.title)}</p>${metrics([['条目通过',result.check_counts.verified || 0],['原始扫描',result.original_count]])}${listTable(['文件','状态','字节'],files)}${notes}`;
}

export function packetReady(packet) {
    const rows = packet.summary.clips.map(clip=>[clip.title,span(clip.source_rows),clip.continuous?.timecodes.join(' – ') || '独立 WAV']);
    return metrics([['监听片段',packet.summary.clip_count],['图像证据',packet.summary.images],['源扫描',packet.summary.source_count]])+
        `<div class="form-footer">${link(`/api/packets/${packet.id}/download`,'下载审听交付 ZIP',{css:'strong'})}${action('verify-retained','校验内容',{id:packet.id})}</div>`+
        (rows.length ? listTable(['监听片段','原片行','串听区间'],rows) : '');
}
