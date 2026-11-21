import {$, action, esc, fmt, image, jsonDetails, link, metrics, scanURL, span} from './ui.js';

export function filmSpan(state) {
    const meta = state.scan.scan.metadata;
    const rows = Math.min(state.frame.rows, meta.height, Math.floor(2150000 / meta.width));
    const a = Math.max(0, Math.min(state.frame.start, meta.height - rows));
    return [a, a + rows];
}
export function filmBox(state) {
    const [a,b] = filmSpan(state);
    return [0,a,state.scan.scan.metadata.width,b];
}
export function currentBox(state) {
    return state.frame.selection || filmBox(state);
}
export function currentPoint(state) {
    const box = currentBox(state);
    return state.frame.point || [Math.floor((box[0] + box[2]) / 2),Math.floor((box[1] + box[3]) / 2)];
}

export function library(state) {
    const reel = state.reel?.reel;
    const here = state.scan?.scan?.id;
    const choices = state.library.reels.map(item => `<button type="button" class="lib-entry" data-action="reel" data-id="${item.id}" aria-current="${item.id === reel?.id}"><span>${item.synthetic ? '<span class="type-dot" aria-label="合成素材">●</span> ' : ''}${esc(item.title)}</span><small class="lib-meta">${esc(item.gauge)} · ${fmt(item.scan_count,0)} 张扫描</small></button>`).join('');
    const scans = (state.reel?.scans || []).map((item,n) => `<button type="button" class="lib-entry" data-action="scan" data-id="${item.id}" aria-current="${item.id === here}"><span><span class="scan-index">${String(n+1).padStart(2,'0')}</span>${item.origin_kind === 'derived_join' ? '<span class="type-dot" title="接续生成">⛓ </span>' : ''}${esc(item.title)}</span><small class="lib-meta">${fmt(item.metadata.height,0)} 行 · ${item.metadata.bit_depth} bit${item.origin_kind === 'derived_join' ? ' · 接续' : ''}</small></button>`).join('');
    return `<h2>档案库</h2><div class="library-list">${choices || '<p class="lib-empty">暂无档案卷</p>'}</div>${reel ? `<h3>原始扫描</h3><div class="library-list">${scans || '<p class="lib-empty">本卷还未导入图像</p>'}</div>${action('intake','导入图像',{},'subtle lib-action')}${action('batch-intake','批量导入 ZIP',{},'quiet lib-action')}` : ''}`;
}

export function blank(state) {
    if (state.reel) {
        return `<div class="empty-reel"><h1>${esc(state.reel.reel.title)}</h1><p class="muted">${esc(state.reel.reel.note || '档案卷已就绪。请导入走片方向沿行延伸的原始声轨边带。')}</p><div class="form-footer">${action('intake','导入图像',{},'strong')}${action('batch-intake','批量导入 ZIP')}</div></div>`;
    }
    return `<section class="empty-state"><h1>把胶片光学声轨读回声音</h1><p>从原始的边带扫描定位透光图案、校准行速、保存监听和图像证据。先载入合成练习，或为你的扫描新建一个档案卷。</p><div class="form-footer">${action('load-demo','载入合成练习',{},'strong')}${action('new-reel','新建档案卷',{},'subtle')}</div></section>`;
}

function terrain(state) {
    const scan = state.scan.scan;
    const document = state.scan.recipe;
    const w = scan.metadata.width, h = scan.metadata.height;
    const [lo,hi] = filmSpan(state);
    const reverse = document.settings.orientation.reverse_y;
    const params = new URLSearchParams({y:lo,bottom:hi,recipe:document.id,view:850,tonal:state.frame.tonal});
    const markTop = Math.min(100,(reverse ? h - hi : lo) / h * 100);
    const chipHeight = (hi-lo) / h * 100;
    const cursor = state.frame.selection ? `已选框  ${state.frame.selection.join(' · ')}` : state.frame.point ? `探针 x ${state.frame.point[0]} · 行 ${state.frame.point[1]}` : '在原片上单击定位，拖动可框选';
    return `<div class="film-view"><div class="film-workarea"><aside class="overview"><b>全幅</b><span>${reverse ? fmt(h,0) : '0'}</span><div id="overview-image" class="overview-image" tabindex="0" role="slider" aria-label="全幅定位" aria-valuemin="0" aria-valuemax="${h-1}" aria-valuenow="${lo}">${image(scanURL(scan.id,`/overview.png?recipe=${document.id}`),'声轨全幅与当前列范围')}<i class="overview-highlight" style="top:${markTop}%;height:${chipHeight}%"></i></div><span>${reverse ? '0' : fmt(h,0)}</span></aside><div class="film-deck"><span class="row-stamp">原片行 ${esc(span([lo,hi]))}</span><div id="film-frame" class="film-frame">${image(scanURL(scan.id,`/tile.png?${params}`),'可定位的原始声轨边带','id="film-image" draggable="false"')}<div class="film-guide track"></div><div class="film-guide perf"></div><div class="film-box" hidden></div></div></div></div><div class="film-controls"><div class="slider-row">${action('frame-step','上一窗',{step:-1})}<input type="range" aria-label="原片起始行" id="film-slider" min="0" max="${Math.max(0,h-(hi-lo))}" value="${lo}" step="1">${action('frame-step','下一窗',{step:1})}</div><div class="film-meta"><span id="film-cursor">${esc(cursor)}</span><div><span class="legend">声轨读窗</span><span class="legend holes">齿孔带</span></div></div></div></div>`;
}

export function filmPage(state) {
    if (!state.scan) return blank(state);
    const scan = state.scan.scan;
    const view = state.scan.recipe;
    const caption = `${scan.metadata.width} 列 × ${fmt(scan.metadata.height,0)} 行`;
    const tools = [action('technical','技术资料'),link(scanURL(scan.id,'/original'),'下载原始扫描'),action('toggle-tonal',state.frame.tonal === 'native' ? '拉伸显示' : '原始亮度')].join('');
    const nav = `${action('window-form','定位行窗')}${action('clear-selection','清除框选')}${action('save-marker','保存定位框')}${action('contact','制作联排')}${action('open-join','接续重叠段')}`;
    const missingPane = state.scan.is_latest ? '' : `<p class="panel-note status-warning">当前正在打开读片历史 v${view.seq}，最新配方为 v${state.scan.latest}。</p>`;
    const derivedPane = state.scan.scan.origin_kind === 'derived_join'
        ? `<p class="panel-note status-good">这是由两张扫描接续生成的连续边带。${action('open-join-record','查看接续来源明细',{},'subtle')}</p>` : '';
    const output = state.result || {html:''};
    return `<header class="scene-header"><div><h1>${esc(scan.title)}</h1><div class="scene-sub"><span>${esc(state.reel.reel.gauge)}</span><span>${esc(caption)}</span><span>${scan.metadata.bit_depth} 位</span><span>配方 v${view.seq}</span>${state.scan.scan.origin_kind === 'derived_join' ? '<span>接续生成</span>' : ''}</div></div><div class="scene-actions">${tools}</div></header>${missingPane}${derivedPane}${terrain(state)}<div class="selection-tools">${nav}</div><section id="result-slot" aria-label="工具结果">${output.html}</section>`;
}

function pos(state, native) {
    const [start,stop] = filmSpan(state);
    const settings = state.scan.recipe.settings;
    const width = state.scan.scan.metadata.width;
    const x = settings.orientation.mirror_x ? width - native[0] : native[0];
    const y = settings.orientation.reverse_y ? stop - native[1] : native[1] - start;
    return [x / width, y / (stop - start)];
}

export function refreshFilmOverlay(state, previewSelection = null) {
    const view = $('#film-frame');
    if (!view || !state.scan) return;
    const recipe = state.scan.recipe.settings;
    for (const [cls, guide] of [['track',recipe.aperture],['perf',recipe.perforations]]) {
        const [left] = pos(state,[guide[0],0]);
        const [right] = pos(state,[guide[1],0]);
        const guideNode = $(`.film-guide.${cls}`,view);
        guideNode.style.left = `${Math.min(left,right)*100}%`;
        guideNode.style.width = `${Math.abs(right-left)*100}%`;
    }
    const selection = previewSelection || state.frame.selection;
    const node = $('.film-box',view);
    node.hidden = !selection;
    node.classList.toggle('drag',Boolean(previewSelection));
    if (selection) {
        const a = pos(state,selection.slice(0,2));
        const b = pos(state,selection.slice(2,4));
        const clip = n => Math.min(1,Math.max(0,n));
        const left = clip(Math.min(a[0],b[0])), right = clip(Math.max(a[0],b[0]));
        const top = clip(Math.min(a[1],b[1])), bottom = clip(Math.max(a[1],b[1]));
        node.style.left = `${left*100}%`;
        node.style.top = `${top*100}%`;
        node.style.width = `${(right-left)*100}%`;
        node.style.height = `${(bottom-top)*100}%`;
        node.hidden = bottom <= top || right <= left;
    }
}

function sourcePoint(state, view, event) {
    const rect = $('#film-image',view).getBoundingClientRect();
    const w = state.scan.scan.metadata.width;
    const [a,b] = filmSpan(state);
    const orientation = state.scan.recipe.settings.orientation;
    const clamp = n => Math.min(1,Math.max(0,n));
    const x = clamp((event.clientX-rect.left)/rect.width);
    const y = clamp((event.clientY-rect.top)/rect.height);
    return [Math.min(w-1,Math.floor((orientation.mirror_x ? 1-x : x)*w)),
        Math.min(b-1,a+Math.floor((orientation.reverse_y ? 1-y : y)*(b-a)))];
}
function bounds(a,b) {
    return [Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.max(a[0],b[0])+1,Math.max(a[1],b[1])+1];
}

export function connectFilm(state, update, selectPoint) {
    const frame = $('#film-frame');
    const mini = $('#overview-image');
    if (!frame || !mini) return;
    refreshFilmOverlay(state);
    let start = null, last = null, origin = null;
    frame.addEventListener('pointerdown', event => {
        if (event.button !== 0) return;
        start = last = sourcePoint(state,frame,event);
        origin = [event.clientX,event.clientY];
        frame.setPointerCapture(event.pointerId);
    });
    frame.addEventListener('pointermove', event => {
        if (!start) return;
        last = sourcePoint(state,frame,event);
        refreshFilmOverlay(state,bounds(start,last));
    });
    frame.addEventListener('pointercancel', () => { start = null; refreshFilmOverlay(state); });
    frame.addEventListener('pointerup', event => {
        if (!start) return;
        const moved = Math.hypot(event.clientX-origin[0],event.clientY-origin[1]);
        if (moved > 4) {
            state.frame.selection = bounds(start,sourcePoint(state,frame,event));
            state.frame.point = null;
        } else {
            state.frame.point = last;
        }
        start = last = null;
        selectPoint();
    });
    const slide = $('#film-slider');
    slide.addEventListener('change', () => update(Number(slide.value)));
    mini.addEventListener('click', event => {
        const position = mini.getBoundingClientRect();
        const ratio = (event.clientY-position.top)/position.height;
        const direction = state.scan.recipe.settings.orientation.reverse_y;
        update(Math.round((direction ? 1-ratio : ratio)*state.scan.scan.metadata.height) - Math.floor(state.frame.rows/2));
    });
    mini.addEventListener('keydown', event => {
        if (!['ArrowUp','ArrowDown','Home','End'].includes(event.key)) return;
        event.preventDefault();
        if (event.key === 'Home') update(0);
        else if (event.key === 'End') update(state.scan.scan.metadata.height-1);
        else update(state.frame.start + (event.key === 'ArrowUp' ? -1 : 1)*state.frame.rows);
    });
}

export function techSummary(report, scan) {
    const info = report.metadata;
    const enc = info.encoding;
    return metrics([['编码',info.format],['扫描码值',`${info.bit_depth} bit`],['图像尺寸',`${info.width} × ${info.height}`],['文件大小',`${fmt(info.file_size/1024)} KiB`]]) + `<p class="panel-note">${esc(info.notices.join(' '))}</p><div class="form-footer">${link(scanURL(scan,'/technical?format=download'),'下载技术资料 JSON')}${link(scanURL(scan,'/original'),'下载原始扫描')}</div>${jsonDetails({encoding:enc,sha256:info.sha256,native_rows:report.witness_rows},'源文件格式与校验')}${jsonDetails(report.profile,'跨列扫描码值分布')}`;
}
