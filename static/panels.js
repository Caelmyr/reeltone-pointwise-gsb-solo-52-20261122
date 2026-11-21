import {$, action, boxInputs, check, checked, dt, esc, field, fmt, form, image, jsonDetails, link, listTable, metrics, number, percent, readBox, record, select, span, submit, TAKE, value} from './ui.js';
import {currentBox, currentPoint, filmSpan} from './film.js';

export const TOOL = {
    exposure:'曝光直方图', crop:'保留原像素证据', contour:'透光轮廓曲线', scope:'监听频谱', perforations:'齿孔间距',
    references:'明暗参照取样', locate:'查找光学读窗', blemish:'局部瑕疵候选', compare:'两张扫描对照',
    sample:'原像素探针', wedge:'阶梯透射标定', splices:'横向接片候选', splice_bands:'多带横向接片',
    slit:'读窗截面漂移', pilot:'基准音稳定度',
};
export const TABS = {recipe:'配方', read:'监听', tools:'巡检', retained:'记录'};
const SIGNAL = [['width','透光宽度'],['light','平均扫描亮度']];
const RATES = [8000,12000,16000,22050,24000,32000,44100,48000].map(n => [n,`${n.toLocaleString()} Hz`]);
const RATE_FIELDS = [['manual','手动行速'],['perfs','齿孔测量']];
const defaults = {mode:'width', rate:24000, gain_db:6, low_hz:40, high_hz:0, ramp_ms:6, center:'median'};
const level = {type:'number', min:0, max:1, step:0.001};
const integerOptions = {type:'number', min:0, step:1};

export function rowsFields(rows, limit) {
    return field('row0','原片起始行',rows[0],{...integerOptions,max:limit-1}) + field('row1','原片停止行',rows[1],{...integerOptions,max:limit});
}
export function extractRows(form) {
    return [number(form,'row0'),number(form,'row1')];
}
export function monitorFields(options) {
    const o = {...defaults,...options};
    return [select('mode','信号测读',SIGNAL,o.mode),select('rate','监听采样率',RATES,o.rate),
        field('gain_db','增益 dB',o.gain_db,{type:'number',min:-24,max:30,step:0.1}),select('center','静音中心',[['median','片段中值'],['mean','片段均值'],['fixed','固定半幅']],o.center),
        field('low_hz','高通 Hz（0 关闭）',o.low_hz,{...integerOptions,max:1600}),
        field('high_hz','低通 Hz（0 关闭）',o.high_hz,{...integerOptions,max:20000}),
        field('ramp_ms','首尾渐变 ms',o.ramp_ms,{type:'number',min:0,max:100,step:0.1})].join('');
}
export function extractMonitor(form) {
    return {mode:value(form,'mode'),rate:number(form,'rate'),center:value(form,'center'),gain_db:number(form,'gain_db'),
        low_hz:number(form,'low_hz'),high_hz:number(form,'high_hz'),ramp_ms:number(form,'ramp_ms')};
}
export function recipeFrom(form) {
    return {aperture:[number(form,'a0'),number(form,'a1')],perforations:[number(form,'p0'),number(form,'p1')],
        orientation:{mirror_x:checked(form,'mirror_x'),reverse_y:checked(form,'reverse_y')},
        tone:{black:number(form,'black'),white:number(form,'white'),threshold:number(form,'threshold'),polarity:value(form,'polarity')},
        clock:{model:value(form,'model'),row_rate:number(form,'row_rate'),frames_per_second:number(form,'frames_per_second'),
            holes_per_frame:number(form,'holes_per_frame'),pitch_guess:number(form,'pitch_guess')}};
}
function recipePanel(state) {
    const d = state.scan.recipe.settings;
    const w = state.scan.scan.metadata.width;
    const controls = [
        field('a0','读窗左列',d.aperture[0],{...integerOptions,max:w-1}),field('a1','读窗右边界',d.aperture[1],{...integerOptions,max:w}),
        field('p0','齿孔带左列',d.perforations[0],{...integerOptions,max:w-1}),field('p1','齿孔带右边界',d.perforations[1],{...integerOptions,max:w}),
        check('mirror_x','左右镜像',d.orientation.mirror_x),check('reverse_y','反向走片',d.orientation.reverse_y),
        field('black','暗参照',d.tone.black,level),field('white','亮参照',d.tone.white,level),
        field('threshold','轮廓阈值',d.tone.threshold,{...level,min:.08,max:.92}),
        select('polarity','轨道极性',[['clear','透光为信号'],['dark','遮光为信号']],d.tone.polarity),
        select('model','时间基准',RATE_FIELDS,d.clock.model,true),
        field('row_rate','手动行／秒',d.clock.row_rate,{type:'number',min:100,max:96000,step:.01}),
        field('pitch_guess','预估孔距（行）',d.clock.pitch_guess,{type:'number',min:8,max:5000,step:.01}),
        field('frames_per_second','每秒画格',d.clock.frames_per_second,{type:'number',min:8,max:60,step:.001}),
        field('holes_per_frame','每格齿孔',d.clock.holes_per_frame,{...integerOptions,min:1,max:12}),
        field('label','版本标签','手工读片校准',{full:true,maxlength:100}),
    ].join('');
    const tools = `${submit('保存读片配方')}${action('apply-selection','用选框定位')}`;
    return `<section class="panel"><div class="panel-title"><h2>读片配方</h2>${action('history','查看版本',{},'subtle')}</div><p class="panel-note">读窗和齿孔带按原片列指定。镜像只改变查看方向，反向走片也会改变监听顺序。</p>${form('recipe-form',controls,tools)}</section>`;
}

function readPanel(state) {
    const h = state.scan.scan.metadata.height;
    const selected = state.frame.selection ? state.frame.selection : [0,0,state.scan.scan.metadata.width,h];
    const rows = state.listenRows || [selected[1],selected[3]];
    const controls = rowsFields(rows,h) + monitorFields(state.monitor) + field('title','监听快照名称','送审监听',{full:true,maxlength:90});
    return `<section class="panel"><h2>监听选段</h2><p class="panel-note">停止行为区间上界。可以先生成试听，再将同一设置保留为 WAV 快照。</p>${form('listen-form',controls,`${submit('生成监听')}${action('save-take','保存监听快照')}${action('listen-selection','采用选框行')}`)}</section>`;
}

function toolControls(state) {
    const t = state.tool;
    const meta = state.scan.scan.metadata;
    const box = currentBox(state);
    const picked = state.frame.selection;
    const h = meta.height;
    const prominentRows = ['perforations','contour','scope','locate','slit'];
    const rows = picked ? [picked[1],picked[3]] : prominentRows.includes(t) ? [0,h] : [box[1],box[3]];
    const smallFields = (items) => items.map(([name,caption,val,min,max,step])=>field(name,caption,val,{type:'number',min,max,step,required:true})).join('');
    const mainBox = boxInputs('box',box,'当前扫描上的原片范围');
    if (t === 'sample') {
        const [x,y] = currentPoint(state);
        return smallFields([['x','原片列',x,0,meta.width-1,1],['y','原片行',y,0,h-1,1],['radius','可见半径',5,2,16,1]]);
    }
    if (t === 'references') {
        const marker = Math.round(meta.width*.63);
        const alt = Math.round(meta.width*.71);
        return boxInputs('dark',[marker,box[1],Math.max(marker+4,Math.floor((marker+alt)/2)-1),box[3]],'暗参照原片框')+
            boxInputs('light',[Math.floor((marker+alt)/2)+1,box[1],Math.max(Math.floor((marker+alt)/2)+5,alt-1),box[3]],'亮参照原片框');
    }
    if (t === 'compare') {
        const choices = state.reel.scans.filter(scan => scan.id !== state.scan.scan.id);
        const first = choices.find(scan=>scan.id === state.pairScan) || choices[0];
        const ow = first?.metadata.width || meta.width;
        const oh = first?.metadata.height || h;
        const optionList = choices.map(scan => [scan.id,scan.title]);
        return select('other_scan','选择同卷的另一张扫描',optionList,first?.id,true)+mainBox+
            boxInputs('other',[0,Math.min(box[1],oh-6),ow,Math.min(oh,Math.max(box[1]+6,box[3]))],'对照扫描上的原片范围');
    }
    if (t === 'exposure' || t === 'crop') return mainBox;
    if (t === 'splices') return mainBox + smallFields([['window','边界一侧样本行',8,3,30,1],['contrast','最小亮度跳变',.08,.015,.9,.001],['spacing','最少间隔行',40,6,1200,1]]);
    if (t === 'splice_bands') return bandFields(state) + smallFields([['window','边界一侧样本行',8,3,30,1],['contrast','最小亮度跳变',.08,.015,.9,.001],['max_slant','最大倾斜（行/列）',.35,.02,1.5,.01]]);
    if (t === 'blemish') return mainBox + smallFields([['contrast','局部反差',.19,.03,.92,.001],['filter_width','邻域宽度',7,3,15,2],['min_pixels','候选像素数',6,2,500,1]]);
    if (t === 'perforations') return rowsFields(rows,h) + smallFields([['pitch_hint','预估孔距',state.scan.recipe.settings.clock.pitch_guess,8,5000,.01]]);
    if (t === 'locate') return rowsFields(rows,h) + smallFields([['width_hint','预期读窗宽',state.scan.recipe.settings.aperture[1] - state.scan.recipe.settings.aperture[0],8,meta.width,1]]);
    if (t === 'scope') return rowsFields(rows,h) + select('resolution','频谱窗口',[['compact','紧凑'],['standard','标准'],['detail','细分']],'standard')+select('take','音源',[['','当前设置'],...state.takes.filter(take=>take.recipe_id===state.scan.recipe.id).map(take=>[take.id,take.title])],'',true);
    if (t === 'slit') return rowsFields(rows,h)+field('pixels_per_mm','每毫米扫描像素数（选填）','',{type:'number',min:1,max:25000,step:.001,required:false,full:true}) + smallFields([['center_deviation','中心偏差像素',3,.2,180,.1],['bins','沿片分箱数',240,20,480,1]]);
    if (t === 'pilot') return rowsFields(rows,h)+smallFields([['frequency','基准音 Hz',455,45,9000,.01],['window_ms','分析窗 ms',140,30,1100,1],['band_percent','搜索半宽 %',7,2,22,.1],['snr','显著度 dB',18,5,50,.1]]);
    if (t === 'wedge') {
        const training = state.scan.scan.metadata.filename === 'ferry-step.png' && state.reel.reel.synthetic;
        const marker = training ? meta.width-20 : Math.max(0,Math.min(meta.width-4,Math.round(meta.width*.63)));
        const right = training ? meta.width-8 : Math.min(meta.width,Math.max(marker+4,Math.round(meta.width*.71)));
        const rows = training ? [.10, .50, .90].map(fraction => Math.round(h*fraction)) : [30,Math.round(h*.34),Math.round(h*.67)];
        return select('baseline','暗端模型',[['fit','估计暗端'],['fixed','以零码值为暗端']],'fit',true)+rows.map((first,n)=>
            boxInputs(`w${n}b`,[marker,first,right,Math.min(first+25,h)],`第 ${n+1} 阶取样框`)+field(`w${n}v`,'已知相对透射值',[.08,.48,.87][n],level)).join('');
    }
    return rowsFields(rows,h);
}

export function toolPayload(state, node) {
    const tool = state.tool;
    const out = {recipe:state.scan.recipe.id};
    if (node.elements.namedItem('row0')) out.rows = extractRows(node);
    if (node.elements.namedItem('box0')) out.box = readBox(node,'box');
    const numeric = ['x','y','radius','window','contrast','spacing','max_slant','filter_width','min_pixels','pitch_hint','width_hint','pixels_per_mm','center_deviation','bins','frequency','window_ms','band_percent','snr'];
    for (const key of numeric) {
        const item = node.elements.namedItem(key);
        if (item && item.value !== '') out[key] = Number(item.value);
    }
    if (tool === 'splice_bands') {
        out.bands = [0,1,2,3].map(n=>readBox(node,`band${n}`)).filter(box=>box[2]>box[0] && box[3]>box[1]);
    }
    if (tool === 'references') {out.dark=readBox(node,'dark');out.light=readBox(node,'light');}
    if (tool === 'compare') {out.other_box=readBox(node,'other');out.other_scan=value(node,'other_scan');}
    if (tool === 'scope') {out.resolution=value(node,'resolution');out.take=value(node,'take') || null;out.monitor={...defaults,...state.monitor};}
    if (tool === 'wedge') {
        out.baseline = value(node,'baseline');
        out.steps = [0,1,2].map(n => ({box:readBox(node,`w${n}b`),target:number(node,`w${n}v`),label:`Step ${n+1}`}));
    }
    if (['pilot','scope'].includes(tool)) out.monitor = {...defaults,...state.monitor};
    return out;
}

function bandFields(state) {
    const meta = state.scan.scan.metadata;
    const recipe = state.scan.recipe.settings;
    const [top, bottom] = filmSpan(state);
    const rows = [top, bottom];
    const picture = Math.max(recipe.aperture[1], recipe.perforations[1]);
    const suggested = [
        [recipe.perforations[0], rows[0], recipe.perforations[1], rows[1]],
        [recipe.aperture[0], rows[0], recipe.aperture[1], rows[1]],
        [picture, rows[0], Math.max(picture + 6, meta.width), rows[1]],
        [0, 0, 0, 0],
    ];
    const chosen = Array.isArray(state.bands) && state.bands.length
        ? [...state.bands, ...suggested].slice(0, 4) : suggested;
    const fields = chosen.map((item,n)=>boxInputs(`band${n}`,item,`观察带 ${n+1}${n>2?'（选填）':''} · 胶片横向位置`)).join('');
    return `<p class="panel-note">用两到四条横跨胶片不同横向位置的观察带找同一条接片线。证据完整的接片须由至少两条带支持，且其中一条不在画格区。各带的行范围取自当前行窗，单条带最多 2,000 行。</p>${fields}`;
}

function toolPanel(state) {
    const selectTool = select('tool','巡检工具',Object.entries(TOOL),state.tool,true);
    const instruction = {compare:'分别指定两张扫描的对应原片范围。',wedge:'在已知透射率的参考片上分别选择三阶均匀区域。',pilot:'选择含有已知频率的参考声段。',splices:'可选取与声轨分开的片边，查找有横向支持的明暗突变。',splice_bands:'在片边、声轨和画格区各圈一条观察带；工具按同一斜线对齐后再判断哪些带确实看见同一处接片。'};
    return `<section class="panel"><h2>原片巡检</h2>${form('tool-form',selectTool+toolControls(state),submit('运行巡检'))}${instruction[state.tool] ? `<p class="panel-note">${esc(instruction[state.tool])}</p>` : ''}</section>`;
}

function retention(state) {
    const takes = state.takes.map(take=>record(take.title,`${TAKE[take.status]} · v${take.payload.parent.recipe_number} · ${fmt(take.payload.audio.duration_seconds)} s`,
        `${action('open-take','打开',{id:take.id})}${link(`/api/takes/${take.id}/wav?download=1`,'WAV')}`)).join('');
    const proofs = state.proofs.map(proof=>record(proof.title,`${proof.category} · ${dt(proof.created)}`,action('open-proof','打开',{id:proof.id}))).join('');
    const markers = state.scan.markers.map(marker=>record(marker.title,`原片 ${marker.display.box.join(' · ')}`,action('open-marker','定位',{id:marker.id}))).join('');
    return `<section class="panel"><div class="panel-title"><h2>监听快照</h2>${state.takes.length > 1 ? action('open-ab','A/B 对听',{},'subtle') : ''}</div><div class="data-list">${takes || '<p class="panel-note">还未保存监听。</p>'}</div></section><section class="panel"><h2>图像证据</h2><div class="data-list">${proofs || '<p class="panel-note">还未保存图像证据。</p>'}</div></section><section class="panel"><h2>定位框</h2><div class="data-list">${markers || '<p class="panel-note">还未保存定位框。</p>'}</div></section>`;
}

export function settingsPanel(state) {
    if (!state.scan) return '';
    const tablist = Object.entries(TABS).map(([id,title])=>`<button type="button" data-action="panel" data-view="${id}" role="tab" aria-selected="${state.panel===id}" aria-controls="panel-inside">${title}</button>`).join('');
    const panels = {recipe:recipePanel,read:readPanel,tools:toolPanel,retained:retention};
    return `<div class="panel-tabs" role="tablist" aria-label="读片台工具">${tablist}</div><div id="panel-inside" role="tabpanel">${panels[state.panel](state)}</div>`;
}

export function auditStubHistory(history) {
    const rows = history.map(recipe=>record(`v${recipe.seq} · ${recipe.label}`,`${dt(recipe.created)} · ${recipe.theme}`,
        `${action('view-recipe','打开',{id:recipe.id})}${action('restore-recipe','复制为新版本',{id:recipe.id})}`));
    return `<div class="data-list">${rows.join('')}</div>`;
}

function readingValue(tool, data) {
    switch (tool) {
        case 'exposure': return [['平均亮度',fmt(data.stats.mean)],['近黑像素',percent(data.stats.near_black_fraction)],['近白像素',percent(data.stats.near_white_fraction)],['读数像素',fmt(data.stats.pixel_count,0)]];
        case 'perforations': return [['代表孔距',data.pitch ? `${fmt(data.pitch)} 行` : '未确认'],['支持孔数',data.trusted_count],['缺孔估计',data.estimated_missing_holes]];
        case 'contour': return [['触读窗的行',fmt(data.summary.touch_window_rows,0)],['多个透光区的行',fmt(data.summary.multiple_island_rows,0)],['轮廓阈值',fmt(data.summary.threshold_absolute)]];
        case 'scope': return [['主频分箱',`${fmt(data.spectrum.dominant_bin_hz,1)} Hz`],['分析带宽',`${fmt(data.spectrum.inspected_band_hz,1)} Hz`],['窗长',`${data.spectrum.window_samples} 样本`]];
        case 'compare': return [['平均亮度差',fmt(data.agreement.mean_delta)],['绝对差 P95',fmt(data.agreement.q95_absolute_delta)],['相关',fmt(data.agreement.linear_r)]];
        case 'blemish': return [['候选',data.candidate_count],['覆盖率',percent(data.coverage,2)],['对比阈值',fmt(data.threshold)]];
        case 'sample': return [['原片坐标',`${data.x}, ${data.y}`],[data.code_description,fmt(data.original_code,0)],['透光读值',fmt(data.decoded_intensity,5)]];
        case 'crop': return [['原片框',data.box.join(' · ')],['码值深度',`${data.sample_bit_depth} bit`]];
        case 'locate': return [['读窗候选',data.candidates.length],['采样行数',fmt(data.samples,0)]];
        case 'references': return [['暗参考',fmt(data.black,5)],['亮参考',fmt(data.white,5)],['透射差',fmt(data.contrast,5)]];
        case 'wedge': return [['参照阶数',data.steps.length],['拟合 γ',fmt(data.response?.gamma)],['拟合误差',fmt(data.response?.robust_error,5)]];
        case 'splices': return [['候选边界',data.candidate_count],['普通透射',fmt(data.ordinary_transmission,5)]];
        case 'splice_bands': return [['辨认出的接片线',data.candidate_count],['证据较完整',data.corroborated_count],['有限证据',data.limited_count],['观察带',data.band_count]];
        case 'slit': return [['有效行',fmt(data.summary.used,0)],['排除行',fmt(data.summary.withheld,0)],['截面中位宽度',`${fmt(data.summary.widths.q10_q50_q90[1],2)} px`]];
        case 'pilot': return [['支持的窗口',`${data.trusted} / ${data.windows}`],['中位频率',`${fmt(data.measured_hz_q10_q50_q90?.[1],3)} Hz`],['频率偏差',`${fmt(data.frequency_drift_ppm,1)} ppm`]];
        default: return [];
    }
}

export function instrumentResult(data) {
    const {kind,reading} = data;
    const img = image(data.preview,`${TOOL[kind]}结果`) + image(data.companion,`${TOOL[kind]}配套图`);
    const warnings = [reading.warning, ...(reading.warnings || []), reading.interpretation, reading.caveat].filter(Boolean);
    const widgets = [data.can_retain ? action('retain-instrument','保存图像证据',{},'strong') : '',
        kind === 'references' && reading.usable ? action('use-references','用于读片配方') : ''].join('');
    const list = kind === 'locate' ? reading.candidates.map((row,n)=>record(`候选 ${n+1} · ${row.guide.join('–')} 列`, `侧向变动评分 ${fmt(row.score)}`,action('apply-track','填入读窗',{a:row.guide[0],b:row.guide[1]}))).join('') : '';
    const bands = kind === 'splice_bands' ? reading.findings.map((line,n)=>spliceLineRecord(line,n)).join('') : '';
    return `<article class="box"><h2>${TOOL[kind]}</h2>${metrics(readingValue(kind,reading))}${warnings.map(line=>`<p class="panel-note status-warning">${esc(line)}</p>`).join('')}<div class="inspect-media">${img}</div><div class="data-list">${list}${bands}</div><div class="form-footer">${widgets}</div>${jsonDetails(reading)}</article>`;
}

const BAND_STATE = {supported:'看见',partial:'只看见一部分',absent:'没看见',outside:'行范围外'};
const LINE_CLASS = {corroborated:'证据较完整',single_band:'只由一条带支持',picture_only:'只在画格区'};
function spliceLineRecord(line, n) {
    const seen = line.observations.filter(item=>item.seen);
    const bands = line.observations.map(item=>`带${item.index+1}${item.role==='picture'?'画格':item.role==='soundtrack'?'声轨':item.role==='perforation'?'齿孔':''}：${BAND_STATE[item.evidence_state]}`).join(' · ');
    const rows = `${fmt(line.line_rows[0],0)} → ${fmt(line.line_rows[1],0)} 行`;
    return record(`线 ${n+1} · ${LINE_CLASS[line.classification]}`,
        `${seen.length}/${line.band_count} 带支持 · 原片行 ${rows} · 斜度 ${fmt(line.slope_rows_per_column,3)} 行/列 · ${bands}`,
        action('line-to-source','回原片核对',{row:line.focus_row,col:line.focus_column}));
}

export function soundResult(report, waveform) {
    const audio = report.audio;
    const metricsList = [['监听时长',`${fmt(audio.duration_seconds,4)} s`],['原片行',span(audio.source_rows)],['采样率',`${fmt(audio.output_rate,0)} Hz`],['原片行速',`${fmt(audio.row_rate,2)} 行/s`]];
    const advisories = [...(audio.warnings || []),...(report.read.warnings || [])];
    const graph = waveform?.length ? '<canvas id="waveform" class="sound-wave" width="940" height="166" role="slider" aria-label="沿波形定位原片行" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0" tabindex="0"></canvas><p class="panel-note">点击波形可回到相应的原片行。音频播放器仍可独立拖动。</p>' : '';
    return `<article class="box"><h2>光学监听结果</h2>${metrics(metricsList)}${graph}${advisories.map(line=>`<p class="panel-note">${esc(line)}</p>`).join('')}${jsonDetails(report.clock,'原片行与时间基准')}</article>`;
}

export function waveCanvas(data, seek) {
    const canvas = $('#waveform');
    if (!canvas || !data || !data.length) return;
    const w = canvas.width, h = canvas.height;
    const context = canvas.getContext('2d');
    context.fillStyle = '#13202a';context.fillRect(0,0,w,h);
    context.strokeStyle = '#36515c';
    for(let y=20;y<h-11;y+=32) {context.beginPath();context.moveTo(0,y);context.lineTo(w,y);context.stroke();}
    const amp = Math.max(.45, Math.min(1.12, Math.max(...data.map(part => Math.max(Math.abs(part.low),Math.abs(part.high))))));
    const y = n => h*.5 - Math.min(1,Math.max(-1,n/amp))*h*.42;
    context.strokeStyle = '#77ddd0';context.lineWidth = Math.max(1,w/data.length);
    context.beginPath();
    for (let n=0;n<data.length;n++) {const x = w*n/data.length;context.moveTo(x,y(data[n].low));context.lineTo(x,y(data[n].high));}
    context.stroke();
    let active = 0;
    function hit(index) {
        active = Math.min(data.length-1,Math.max(0,index));
        canvas.setAttribute('aria-valuenow',String(Math.round(active/(data.length||1)*100)));
        seek(data[active]);
    }
    canvas.addEventListener('click',event=>{const box = canvas.getBoundingClientRect();hit(Math.floor((event.clientX-box.left)/box.width*data.length));});
    canvas.addEventListener('keydown',event=>{if (!['ArrowLeft','ArrowRight'].includes(event.key)) return;event.preventDefault();hit(active+(event.key==='ArrowLeft'?-3:3));});
}

export function readingDetails(verification) {
    const {record:take,source,recipe,verified,integrity} = verification;
    const report = take.payload;
    const rate = report.audio.output_rate;
    return metrics([['复核状态',TAKE[take.status]],['配方',`v${recipe.seq}`],['时长',`${fmt(report.audio.duration_seconds,4)} s`],['采样率',`${fmt(rate,0)} Hz`]]) +
        `<p class="panel-note ${verified?'status-good':'status-error'}">${verified?'源图、配方和 WAV 摘要均通过校验。':'保存的证据有差异。'}</p><audio controls preload="metadata" src="/api/takes/${take.id}/wav" style="width:100%"></audio><div class="form-footer">${link(`/api/takes/${take.id}/wav?download=1`,'下载 WAV')}${action('set-take','标为已复核',{id:take.id,state:'reviewed'},'strong')}${action('set-take','暂缓',{id:take.id,state:'hold'})}${action('set-take','转回草稿',{id:take.id,state:'draft'})}${action('take-to-source','定位原片',{id:source.id,recipe:recipe.id,row:report.audio.source_rows[0]})}${action('open-replay','复算核对',{id:take.id})}</div>${jsonDetails(integrity,'校验明细')}`;
}
