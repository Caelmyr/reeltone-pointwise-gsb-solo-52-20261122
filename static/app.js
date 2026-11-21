import {$, all, action, boxInputs, check, checked, closeModal, dt, esc, field, fmt, form, image, jsonDetails, link, listTable, metrics, modal, notice, number, post, readBox, record, reelURL, request, scanURL, select, shorten, span, submit, textarea, value} from './ui.js';
import {blank, connectFilm, currentBox, filmPage, filmSpan, library, refreshFilmOverlay, techSummary} from './film.js';
import {auditStubHistory, extractMonitor, extractRows, instrumentResult, readingDetails, recipeFrom, referencePins, rowsFields, settingsPanel, soundResult, TABS, TOOL, toolPayload, waveCanvas} from './panels.js';
import {cueForm, deskPage, packetReady, zipSummary} from './desk.js';

const state = {
    library:{reels:[]}, reel:null, scan:null, takes:[], proofs:[], workspace:'scan', panel:'recipe', tool:'exposure',
    frame:{start:0,rows:480,tonal:'native',point:null,selection:null}, monitor:{}, listenRows:null, result:null,
    instrumentContext:null, pairScan:null, refMode:'off', refBoxes:{}, desk:null, modalData:null, scanTicket:0,
};

function drawLibrary() { $('#library').innerHTML = library(state); }
function drawSettings() { $('#settings').innerHTML = settingsPanel(state); }
function drawWorkspace() {
    const root = $('#workbench');
    root.classList.toggle('desk',state.workspace==='desk');
    for (const button of all('.global-tabs [data-view]')) button.setAttribute('aria-pressed', String(button.dataset.view===state.workspace));
    $('#workspace').innerHTML = state.workspace==='desk' ? deskPage(state) : filmPage(state);
    if (state.workspace==='scan' && state.scan) {
        connectFilm(state,windowStart,onFilmSelection);
        waveCanvas(state.result?.waveform,bin=>windowStart(Math.floor(bin.source_edges[0])-Math.floor(state.frame.rows/2)));
    }
}
function draw() { drawLibrary(); drawWorkspace(); drawSettings(); }
function onFilmSelection() {
    refreshFilmOverlay(state);
    const message = state.frame.selection ? `已选框 ${state.frame.selection.join(' · ')}` : `探针 x ${state.frame.point[0]} · 行 ${state.frame.point[1]}`;
    $('#film-cursor').textContent = message;
    if (state.panel==='tools' || state.panel==='read') drawSettings();
}
function windowStart(start) {
    state.frame.start = start;
    state.frame.start = filmSpan(state)[0];
    drawWorkspace();
}
function transport(audio) {
    const current = $('#transport');
    if (!audio) {
        current.innerHTML = '';
        current.hidden = true;
        return;
    }
    current.hidden = false;
    current.innerHTML = `<div><b>${esc(audio.title)}</b><small>${esc(audio.detail)}</small></div><audio controls preload="metadata" src="${esc(audio.url)}" aria-label="当前监听音频"></audio><div class="form-footer">${link(audio.url,'下载 WAV',{download:audio.name || 'reeltone-audition.wav'})}${action('close-transport','关闭')}</div>`;
}
function play(data) {
    transport(data);
}
function result(html, waveform = null) {
    state.result = {html,waveform};
    drawWorkspace();
}
function deskDefaults() {
    return {takes:[],proofs:[],cues:[],inventory:null,items:[],checkedProofs:[],title:'档案监听包',continuous:true,include_cues:true,include_sources:false,rate:24000};
}

async function loadWorkspace() {
    state.library = await request('/api/library');
    if (state.library.reels.length) await openReel(state.library.reels[0].id);
    else draw();
}

async function openReel(id) {
    const document = await request(reelURL(id));
    state.reel = document;
    state.desk = null;
    state.scan = null;
    state.takes = [];
    state.proofs = [];
    state.panel = 'recipe';
    state.result = null;
    if (document.scans.length) await openScan(document.scans[0].id);
    if (state.workspace==='desk') await loadDesk();
    else draw();
}

async function openScan(id, recipe = null, {keepPosition = false, workspace = false} = {}) {
    const ticket = ++state.scanTicket;
    const extra = recipe ? `?recipe=${recipe}` : '';
    const [source, takes, proofs] = await Promise.all([
        request(scanURL(id,extra)),request(scanURL(id,'/takes')),request(scanURL(id,'/proofs')),
    ]);
    if (ticket!==state.scanTicket) return;
    state.scan = source;
    state.takes = takes.takes;
    state.proofs = proofs.proofs;
    if (workspace) state.workspace='scan';
    if (!keepPosition) {
        state.frame.start = 0;
        state.frame.selection = null;
        state.frame.point = null;
        state.listenRows = null;
        state.result = null;
        state.instrumentContext = null;
        state.pairScan = null;
    }
    draw();
}

async function loadDesk() {
    if (!state.reel) return;
    const reel = state.reel.reel.id;
    const scans = state.reel.scans;
    if (!state.desk) state.desk = deskDefaults();
    const draft = state.desk;
    const records = Promise.all(scans.map(async scan=>{
        const [take, proof] = await Promise.all([request(scanURL(scan.id,'/takes')),request(scanURL(scan.id,'/proofs'))]);
        return [...take.takes.map(item=>['take',item]),...proof.proofs.map(item=>['proof',item])];
    }));
    const [doc, cues, inventory, allRecords] = await Promise.all([
        request(reelURL(reel)),request(reelURL(reel,'/cues')),request(reelURL(reel,'/readiness')),records,
    ]);
    if (state.reel?.reel.id!==reel || state.desk!==draft) return;
    state.reel = doc;
    draft.cues = cues.cues;
    draft.inventory = inventory;
    const flat = allRecords.flat();
    draft.takes = flat.filter(item=>item[0]==='take').map(item=>item[1]);
    draft.proofs = flat.filter(item=>item[0]==='proof').map(item=>item[1]);
    draw();
}

function updatePacketForm() {
    const f = $('#packet-form');
    if (!f || !state.desk) return;
    for (const name of ['title','continuous','include_cues','include_sources','rate']) {
        const elm = f.elements.namedItem(name);
        state.desk[name] = elm.type==='checkbox' ? elm.checked : name==='rate' ? Number(elm.value) : elm.value;
    }
    state.desk.checkedProofs = all('[name="proof"]:checked',f).map(entry=>entry.value);
    for (const row of all('.playlist-item',f)) {
        const item = state.desk.items[Number(row.dataset.item)];
        const past = item.take;
        item.take = $('[name="playtake"]',row).value;
        item.cue = past===item.take ? $('[name="playcue"]',row).value : '';
        item.gap_ms = Number($('[name="playgap"]',row).value);
    }
}
function rerenderPlaylist() {
    drawWorkspace();
}

function makeReelModal() {
    state.modalData=null;
    const input = field('title','档案卷标题','',{full:true,maxlength:120})+
        field('accession','入藏编号','',{maxlength:70,required:false})+
        select('gauge','胶片规格',[['16 mm','16 mm'],['35 mm','35 mm'],['Other','其他']],'16 mm')+
        textarea('note','档案备注','',800);
    modal('新建档案卷',form('reel-form',input,submit('创建档案卷')));
}
function intakeModal() {
    state.modalData = {reel:state.reel.reel.id};
    const details = field('scan','扫描文件','',{type:'file',accept:'image/png,image/tiff,image/jpeg,.tif,.tiff',full:true})+
        field('title','扫描标题（选填）','',{full:true,required:false,maxlength:110})+
        textarea('note','来源说明','',700);
    modal('导入一张声轨边带',`<p class="panel-note">接受 PNG、单页 TIFF 和 JPEG。声轨长度沿行向下展开。未改动的上传字节会作为原档保存。</p>${form('intake-form',details,`${submit('导入扫描')}${action('intake-preflight','先检查图像')}`)}<div id="intake-result"></div>`);
}
function batchIntakeModal() {
    state.modalData={reel:state.reel.reel.id,digest:null};
    const fields=field('batch','扫描 ZIP','',{type:'file',accept:'.zip,application/zip',full:true});
    modal('批量导入边带扫描',`<p class="panel-note">ZIP 内放最多 14 张 PNG、单页 TIFF 或 JPEG。可附 intake.json 标注来源与走片方向。</p>${form('batch-form',fields,`${submit('预检扫描批次')}${action('commit-batch','确认导入已检查批次')}`)}<div id="intake-result"></div>`);
}

function batchSummary(result) {
    const list = result.files.map(item=>[item.path,`${item.metadata.width} × ${item.metadata.height}`,item.byte_duplicate ? '已在卷内' : item.duplicates_inside_batch.length ? '本批重复' : '待导入']);
    return metrics([['准备导入',result.ready_to_add],['扫描成员',result.entries]])+listTable(['文件','原片像素','预检'],list)+`<div class="preview-grid">${result.files.filter(item=>item.preview).map(item=>`<div class="preview-card">${image(item.preview,item.title)}<small>${esc(item.title)}</small></div>`).join('')}</div>`;
}
async function batchRead(node, actionName) {
    if (!node.reportValidity()) return;
    const draft = state.modalData;
    if (actionName==='commit' && !draft.digest) throw Error('先预检要导入的 ZIP。');
    const transfer = new FormData(node);
    transfer.append('action',actionName);
    if (draft.digest) transfer.append('sha256',draft.digest);
    const checked = await post(reelURL(draft.reel,'/scan-batch'),transfer);
    if (actionName==='commit') {
        state.reel = await request(reelURL(draft.reel));
        const first = checked.imported.find(item=>item.created)?.scan || checked.imported[0]?.scan;
        closeModal();
        if (first) await openScan(first.id, null, {workspace:true});
        state.library = await request('/api/library');drawLibrary();
        notice(`导入完成，新增 ${checked.added_count} 张。`);
    } else {
        draft.digest = checked.sha256;
        $('#intake-result').innerHTML=batchSummary(checked);
    }
}

async function retainTake(formNode, save) {
    if (!formNode || !formNode.reportValidity()) return;
    const source = state.scan;
    state.monitor = extractMonitor(formNode);
    state.listenRows = extractRows(formNode);
    const title = value(formNode,'title');
    const payload = {title, rows:state.listenRows,monitor:state.monitor,recipe:source.recipe.id,retain:save};
    const completed = await post(scanURL(source.scan.id,'/audition'),payload);
    if (state.scan?.scan.id!==source.scan.id) return;
    const reading = save ? completed.take.payload : completed.report;
    const url = save ? `/api/takes/${completed.take.id}/wav` : completed.audio;
    play({title,detail:`${fmt(reading.audio.duration_seconds,4)} s · ${fmt(reading.audio.output_rate,0)} Hz`,url,userCreated:save});
    result(soundResult(reading,completed.waveform),completed.waveform);
    if (save) {
        state.takes = (await request(scanURL(source.scan.id,'/takes'))).takes;
        notice('已保存监听快照。可在「记录」内复核。');
    }
}

function viewProof(resultData) {
    const info = resultData.proof;
    const primary = `/api/proofs/${info.id}/primary.png`;
    const secondary = `/api/proofs/${info.id}/secondary.png`;
    const second = info.secondary_sha256 ? image(secondary,'配套证据图') : '';
    const parentBox = info.evidence.box || info.evidence.region;
    const near = Array.isArray(parentBox) ? parentBox[1] : (info.evidence.source_rows?.[0] || 0);
    const tools = [link(primary+'?download=1','下载证据 PNG'), info.secondary_sha256 ? link(secondary+'?download=1','下载配套 PNG') : '',
        link(`/api/proofs/${info.id}/reading.json`,'下载来源与读数 JSON'),action('take-to-source','定位来源',{id:info.scan_id,recipe:info.recipe_id,row:near})].join('');
    const pins = info.category === 'comparison' ? referencePins(info.evidence, 'compare') : '';
    modal(info.title,`<p class="panel-note ${resultData.verified?'status-good':'status-error'}">${resultData.verified?'内容和源位置校验通过。':'保存内容在校验中出现差异。'}</p><div class="inspect-media">${image(primary,'留存证据图')}${second}</div>${pins ? `<p class="panel-note">留存的两张参照框仍可点回各自的扫描核对；校正行带状态见读数 JSON。</p>${pins}` : ''}<div class="form-footer">${tools}</div>${jsonDetails(info.evidence)}${jsonDetails(resultData.integrity,'校验核对')}`);
}

function originalJump(data) {
    closeModal();
    return openScan(data.id,data.recipe || null,{workspace:true}).then(()=>windowStart(Number(data.row || 0)));
}

function openContact() {
    const w = state.scan.scan.metadata.width,h=state.scan.scan.metadata.height;
    const active = currentBox(state);
    const choices = [active,[active[0],Math.max(0,Math.min(h-8,active[3]+2)),active[2],Math.min(h,Math.max(active[3]+8,active[3]+(active[3]-active[1])))]];
    choices[1][3] = Math.max(choices[1][1]+8,choices[1][3]);
    let content = field('title','联排名称','Film contact sheet',{full:true,maxlength:85})+
        select('paper','纸张',[['A4','A4'],['Letter','Letter']],'A4')+select('layout','纸张方向',[['portrait','竖排'],['landscape','横排']],'portrait');
    for (let i=0;i<choices.length;i++) {
        content += check(`enable${i}`,`纳入第 ${i+1} 格`,i===0,true)+field(`panel${i}`,`第 ${i+1} 格标签`,`Source ${i+1}`,{full:true,maxlength:65})+boxInputs(`panelbox${i}`,choices[i]);
    }
    modal('制作原片联排',form('contact-form',content,submit('生成 PNG 与打印 PDF')));
}

function jumpCue(id, edit) {
    const cue = state.desk?.cues.find(item=>item.id===id);
    if (!cue) return;
    return openScan(cue.scan_id,null,{workspace:!edit}).then(()=>{
        if (edit) {state.modalData={cue};modal('编辑审听标记',cueForm(state,cue));}
        else windowStart(cue.y0);
    });
}

const commands = {
    'load-demo':async () => {
        const response=await post('/api/training');
        state.library=await request('/api/library');
        state.workspace='scan';
        await openReel(response.training[0].reel.id);
        notice(`已载入 ${response.scan_count} 张合成扫描。`);
    },
    'new-reel':makeReelModal,
    'reel':data=>openReel(data.id),
    'scan':data=>openScan(data.id,null,{workspace:true}),
    'close-modal':closeModal,
    'close-transport':()=>play(null),
    'workspace':async data=>{state.workspace=data.view;drawWorkspace();if(data.view==='desk') await loadDesk();},
    'panel':data=>{state.panel=data.view;drawSettings();},
    'intake':intakeModal,
    'batch-intake':batchIntakeModal,
    'commit-batch':()=>batchRead($('#batch-form'),'commit'),
    'intake-preflight':async ()=>{
        const node=$('#intake-form');if(!node.reportValidity()) return;
        const answer=await post(reelURL(state.modalData.reel,'/intake'),new FormData(node));
        const check=[['图像编码',answer.metadata.format],['图像大小',`${answer.metadata.width} × ${answer.metadata.height}`],['重复检查',answer.duplicate?'卷内已有':'可导入']];
        $('#intake-result').innerHTML=metrics(check)+image(answer.preview,'导入图像全幅预检','style="max-height:215px"');
    },
    'frame-step':data=>windowStart(state.frame.start+Number(data.step)*state.frame.rows),
    'toggle-tonal':()=>{state.frame.tonal=state.frame.tonal==='native'?'stretch':'native';drawWorkspace();},
    'clear-selection':()=>{state.frame.selection=null;state.frame.point=null;state.listenRows=null;drawWorkspace();drawSettings();},
    'window-form':()=>{
        const box=filmSpan(state),h=state.scan.scan.metadata.height;
        modal('定位原片行窗',form('window-form',field('start','首行',box[0],{type:'number',min:0,max:h-1,step:1})+field('rows','显示行数',box[1]-box[0],{type:'number',min:32,max:Math.min(2600,h),step:1}),submit('定位')));
    },
    'technical':async ()=>{const id=state.scan.scan.id;const tech=await request(scanURL(id,`/technical?recipe=${state.scan.recipe.id}`));modal('源扫描技术资料',techSummary(tech,id));},
    'save-marker':()=>modal('保存定位框',form('marker-form',field('title','定位框名称','',{full:true,maxlength:85})+
        select('category','定位类型',[['track','声轨'],['reference','参照'],['splice','接片'],['surface','片面']],'track',true)+
        boxInputs('box',currentBox(state),'原片位置')+textarea('note','说明','',600),submit('保存定位框'))),
    'open-marker':async data=>{
        const found=await post(scanURL(state.scan.scan.id,'/marker-view'),{marker:data.id,recipe:state.scan.recipe.id});
        const box=found.source_box;state.frame.selection=box;windowStart(box[1]);closeModal();notice(`已定位「${found.marker.title}」。`);
    },
    'history':async ()=>modal('读片历史',auditStubHistory((await request(scanURL(state.scan.scan.id,'/recipes'))).recipes)),
    'view-recipe':data=>{closeModal();return openScan(state.scan.scan.id,data.id,{keepPosition:true});},
    'restore-recipe':async data=>{const scan=state.scan;const next=await post(scanURL(scan.scan.id,`/recipes/${data.id}/restore`),{expected:scan.latest});closeModal();await openScan(scan.scan.id,next.recipe.id,{keepPosition:true});notice(`已复制为配方 v${next.recipe.seq}。`);},
    'apply-selection':()=>{
        if (!state.frame.selection) throw Error('在原片上拖动框选读窗。');
        const [left,,right] = state.frame.selection;
        const n=$('#recipe-form');n.elements.a0.value=left;n.elements.a1.value=right;
    },
    'listen-selection':()=>{
        if (!state.frame.selection) throw Error('先在原片上框选一段扫描。');
        state.listenRows=[state.frame.selection[1],state.frame.selection[3]];drawSettings();
    },
    'save-take':()=>retainTake($('#listen-form'),true),
    'open-take':async data=>{const got=await request(`/api/takes/${data.id}`);modal(got.record.title,readingDetails(got));},
    'open-proof':async data=>viewProof(await request(`/api/proofs/${data.id}`)),
    'set-take':async data=>{
        const out=await post(`/api/takes/${data.id}/status`,{status:data.state});
        if (state.scan.scan.id===out.take.scan_id) state.takes=(await request(scanURL(out.take.scan_id,'/takes'))).takes;
        const got=await request(`/api/takes/${data.id}`);drawSettings();modal(out.take.title,readingDetails(got));
        if(state.workspace==='desk') await loadDesk();
    },
    'take-to-source':originalJump,
    'open-native':originalJump,
    'apply-track':data=>{state.panel='recipe';drawSettings();$('#recipe-form').elements.a0.value=data.a;$('#recipe-form').elements.a1.value=data.b;notice('已将候选填入配方。核对列范围后保存。');},
    'use-references':()=>{const targets=state.instrumentContext?.response.reading;if(!targets?.usable) return;state.panel='recipe';drawSettings();$('#recipe-form').elements.black.value=targets.black;$('#recipe-form').elements.white.value=targets.white;},
    'ref-use':data=>{
        if (!state.frame.selection) throw Error('先在原片上拖动框选一个参照区。');
        if (state.scan?.scan?.id!==data.scan) throw Error('请先在左侧打开这张扫描，再框选参照后填入。');
        const form=$('#tool-form');
        if (!form) throw Error('请先回到「巡检」的对照工具。');
        if (data.box.includes('2') && value(form,'other_scan')!==data.scan) {
            throw Error('当前打开的是第二张扫描。请把「选择同卷的另一张扫描」选成第一张，再运行对照。');
        }
        ['0','1','2','3'].forEach((suffix,i)=>{const input=form.elements.namedItem(`${data.box}${suffix}`);if(input) input.value=state.frame.selection[i];});
        notice(`已把当前选框填入${data.box.includes('2')?'第二张':'第一张'}的参照框。`);
    },
    'show-reference':data=>{
        state.workspace='scan';
        const jump=()=>{
            const box=String(data.box||'').split(',').map(Number);
            if (box.length===4 && box.every(Number.isFinite)) {state.frame.selection=box;windowStart(box[1]);}
            notice(`已定位「${data.title||'明暗参照'}」框。`);
        };
        if (state.scan?.scan?.id===data.scan) {jump();return Promise.resolve();}
        return openScan(data.scan,data.recipe || null,{workspace:true}).then(jump);
    },
    'retain-instrument':()=>{
        if(!state.instrumentContext?.response.can_retain) return;
        const tool=state.instrumentContext.kind;
        modal('保存巡检证据',form('proof-form',field('title','证据名称',TOOL[tool],{full:true,maxlength:90})+textarea('note','备注','',650),submit('保存图像证据')));
    },
    'contact':openContact,
    'open-ab':()=>{
        const takes=state.takes.map(take=>[take.id,take.title]);
        modal('A/B 监听核对',form('ab-form',select('a','监听 A（左）',takes,takes[0][0])+select('b','监听 B（右）',takes,takes[1][0])+
            select('gain','响度匹配',[['none','保留原样'],['rms','均方能量'],['percentile','峰值分位']],'none')+field('cap','增益调整上限 dB',9,{type:'number',min:0,max:24,step:.1}),submit('对听共同原片范围')));
    },
    'open-replay':data=>{
        state.modalData={take:data.id};modal('复算已存监听',form('replay-form',select('recipe','复算版本',[['retained','保存监听时的配方'],['current','当前最新配方']],'retained',true)+check('fades','略过首尾渐变',false,true)+field('tolerance','样本差异告警阈值',0.002,{type:'number',min:.00001,max:.2,step:.00001,full:true}),submit('复算并核对')));
    },
    'new-cue':()=>{state.modalData={cue:null};modal('添加审听标记',cueForm(state));},
    'edit-cue':data=>jumpCue(data.id,true), 'cue-view':data=>jumpCue(data.id,false),
    'view-readiness':()=>modal('档案卷覆盖图',image(reelURL(state.reel.reel.id,'/readiness/png'),'审听与原片扫描范围覆盖')),
    'item-add':()=>{updatePacketForm();if(state.desk.items.length>=12) throw Error('最多可排列 12 份监听。');state.desk.items.push({take:'',cue:'',gap_ms:150});rerenderPlaylist();},
    'item-remove':data=>{updatePacketForm();state.desk.items.splice(Number(data.n),1);rerenderPlaylist();},
    'item-up':data=>shiftItem(Number(data.n),-1), 'item-down':data=>shiftItem(Number(data.n),1),
    'verify-zip':()=>modal('核验收到的 ZIP',form('verify-form',field('packet','交付 ZIP','',{type:'file',accept:'.zip,application/zip',full:true}),submit('核验文档和来源'))),
    'verify-retained':async data=>modal('已保存交付包校验',zipSummary(await request(`/api/packets/${data.id}/verify`))),
    'events':async data=>{
        const reel=state.reel.reel.id;
        const answer=await request(reelURL(reel,`/events${data.before?`?before=${data.before}`:''}`));
        const logs=answer.events.map(event=>[dt(event.time),event.kind,event.detail.title || event.detail.label || shorten(event.subject_id)]);
        modal('档案变更记录',listTable(['时间','操作','对象'],logs)+(answer.next?`<div class="form-footer">${action('events','更早的记录',{before:answer.next})}</div>`:''));
    },
};

function shiftItem(at, by) {
    updatePacketForm();
    const target = at+by;
    if (target<0 || target>=state.desk.items.length) return;
    const [entry] = state.desk.items.splice(at,1);
    state.desk.items.splice(target,0,entry);
    rerenderPlaylist();
}

const submitters = {
    'reel-form':async node=>{
        const response=await post('/api/reels',{title:value(node,'title'),accession:value(node,'accession'),gauge:value(node,'gauge'),note:value(node,'note')});
        state.library=await request('/api/library');closeModal();await openReel(response.reel.id);notice('档案卷已建立。');
    },
    'intake-form':async node=>{
        const id=state.modalData.reel;
        const answer=await post(reelURL(id,'/scans'),new FormData(node));
        state.reel=await request(reelURL(id));state.library=await request('/api/library');closeModal();await openScan(answer.scan.id,null,{workspace:true});
        notice(answer.duplicate?'扫描在本卷已有记录，已打开原条目。':'已保存原始扫描。');
    },
    'batch-form':node=>batchRead(node,'preflight'),
    'window-form':node=>{state.frame.rows=number(node,'rows');closeModal();windowStart(number(node,'start'));drawSettings();},
    'recipe-form':async node=>{
        const scan=state.scan;
        const next=await post(scanURL(scan.scan.id,'/recipe'),{settings:recipeFrom(node),label:value(node,'label'),expected:scan.latest});
        await openScan(scan.scan.id,next.recipe.id,{keepPosition:true});
        notice(next.created?`已保存配方 v${next.recipe.seq}。`:`设置与 v${next.recipe.seq} 相同。`);
    },
    'listen-form':node=>retainTake(node,false),
    'tool-form':async node=>{
        const scan=state.scan.scan.id;
        const kind=state.tool;
        const payload=toolPayload(state,node);
        const response=await post(scanURL(scan,`/instrument/${kind}`),payload);
        if (state.scan.scan.id!==scan) return;
        state.instrumentContext={scan,kind,payload,response};
        result(instrumentResult(response));
        $('#result-slot').scrollIntoView({block:'nearest'});
    },
    'proof-form':async node=>{
        const details=state.instrumentContext;
        const completed=await post(scanURL(details.scan,`/instrument/${details.kind}`),{...details.payload,retain:true,title:value(node,'title'),note:value(node,'note')});
        state.proofs=(await request(scanURL(details.scan,'/proofs'))).proofs;
        const inspected=await request(`/api/proofs/${completed.proof.id}`);
        viewProof(inspected);drawSettings();notice('已保存图像证据和来源读数。');
    },
    'marker-form':async node=>{
        const scan=state.scan;
        const marker=await post(scanURL(scan.scan.id,'/markers'),{title:value(node,'title'),category:value(node,'category'),note:value(node,'note'),box:readBox(node),recipe:scan.recipe.id});
        state.scan.markers.unshift(marker.marker);closeModal();drawSettings();notice('定位框已保存。');
    },
    'contact-form':async node=>{
        const panels=[0,1].filter(index=>checked(node,`enable${index}`)).map(index=>({box:readBox(node,`panelbox${index}`),title:value(node,`panel${index}`),tonal:state.frame.tonal}));
        const final=await post(scanURL(state.scan.scan.id,'/contact'),{recipe:state.scan.recipe.id,panels,title:value(node,'title'),paper:value(node,'paper'),layout:value(node,'layout')});
        modal('原片联排预览',image(final.preview,'原片联排预览','style="max-height:560px;object-fit:contain"')+`<div class="form-footer">${link(final.pdf,'下载打印 PDF',{download:'film-contact.pdf',css:'strong'})}${link(final.preview,'下载联排 PNG',{download:'film-contact.png'})}</div>`+jsonDetails(final.reading));
    },
    'ab-form':async node=>{
        const output=await post(scanURL(state.scan.scan.id,'/ab'),{a:value(node,'a'),b:value(node,'b'),gain:value(node,'gain'),cap:number(node,'cap')});
        const r=output.report;
        const display=metrics([['共同原片行',span(r.native_rows)],['样本相关',fmt(r.statistics.correlation,6)],['右声部电平倍率',fmt(r.right_gain,5)]])+image(output.preview,'A/B 原片位置的能量对比')+`<audio controls src="${esc(output.audio)}" style="width:100%"></audio><div class="form-footer">${link(output.audio,'下载双耳 WAV',{download:'reeltone-ab.wav'})}</div>`+jsonDetails(r);
        modal('A/B 监听结果',display);
    },
    'replay-form':async node=>{
        const take=state.modalData.take;
        const answer=await post(`/api/takes/${take}/replay`,{recipe:value(node,'recipe'),ignore_fades:checked(node,'fades'),tolerance:number(node,'tolerance')});
        const r=answer.result;
        modal('监听复算结果',metrics([['文件字节',r.identical_wav?'相同':'有差异'],['样本对比数',fmt(r.comparison.compared_samples,0)],['最大样本差',fmt(r.comparison.max_absolute_difference,7)]])+
            r.warnings.map(item=>`<p class="panel-note">${esc(item)}</p>`).join('')+image(answer.preview,'样本复算误差图')+
            `<div class="form-footer">${link(answer.audio,'下载复算 WAV',{download:'replay.wav'})}</div>${jsonDetails(r)}`);
    },
    'cue-form':async node=>{
        const existing=state.modalData?.cue;
        const fields={title:value(node,'title'),rows:extractRows(node),kind:value(node,'kind'),status:value(node,'status'),take:value(node,'take') || null,note:value(node,'note')};
        if(existing) fields.previous_updated=existing.updated;
        await request(scanURL(state.scan.scan.id,`/cues${existing?`/${existing.id}`:''}`),{method:existing?'PUT':'POST',data:fields});
        closeModal();if(state.workspace==='desk') await loadDesk();else await commands.workspace({view:'desk'});notice('审听标记已保存。');
    },
    'packet-form':async ()=>{
        updatePacketForm();
        const d=state.desk;
        if(d.items.some(item=>!item.take)) throw Error('请选择每个顺序项的监听，或移除空行。');
        const result=await post(reelURL(state.reel.reel.id,'/packets'),{title:d.title,items:d.items,proofs:d.checkedProofs,
            include_sources:d.include_sources,continuous:d.continuous,include_cues:d.include_cues,rate:d.rate});
        await loadDesk();modal(result.packet.title,packetReady(result.packet));notice('档案交付包已保存。');
    },
    'verify-form':async node=>modal('交付 ZIP 校验结果',zipSummary(await post('/api/packets/verify',new FormData(node)))),
};

async function handle(work, target) {
    if(target?.disabled) return;
    if(target) target.disabled=true;
    try { await work(); }
    catch (error) { notice(error.message || '操作遇到问题。',true); }
    finally { if(target) target.disabled=false; }
}

document.addEventListener('click',event=>{
    const button=event.target.closest('button[data-action]');
    if(!button) return;
    const work=commands[button.dataset.action];
    if(work) handle(()=>work({...button.dataset}),button);
});

document.addEventListener('submit',event=>{
    const node=event.target;
    const handler=submitters[node.id];
    if(!handler) return;
    event.preventDefault();
    if(!node.reportValidity()) return;
    handle(()=>handler(node),event.submitter || $('button[type="submit"]',node));
});

document.addEventListener('change',event=>{
    const node=event.target;
    const form=node.form;
    if(form?.id==='tool-form' && node.name==='tool') {state.tool=node.value;drawSettings();}
    if(form?.id==='tool-form' && node.name==='other_scan') {
        if (form.elements.namedItem('box0')) state.pairBoxes={first:readBox(form,'box'),other:readBox(form,'other')};
        state.pairScan=node.value;drawSettings();
    }
    if(form?.id==='tool-form' && node.name==='refs_on') {
        if(node.checked && form.elements.namedItem('refdark0')) {
            const scanId=state.scan.scan.id;
            const otherId=state.pairScan || state.reel.scans.find(scan=>scan.id!==scanId)?.id;
            if(otherId) state.refBoxes={...(state.refBoxes||{}),
                [scanId]:{dark:readBox(form,'refdark'),light:readBox(form,'reflight')},
                [otherId]:{dark:readBox(form,'refdark2'),light:readBox(form,'reflight2')}};
        }
        state.refMode=node.checked?'on':'off';drawSettings();
    }
    if(form?.id==='batch-form' && node.name==='batch') {state.modalData.digest=null;$('#intake-result').replaceChildren();}
    if(form?.id==='packet-form') {updatePacketForm();if(node.name==='playtake') rerenderPlaylist();}
});

document.addEventListener('keydown',event=>{
    if(event.target.getAttribute('role')!=='tab' || !['ArrowLeft','ArrowRight'].includes(event.key)) return;
    event.preventDefault();
    const options=Object.keys(TABS);
    const index=options.indexOf(state.panel);
    state.panel=options[(index+(event.key==='ArrowLeft'?-1:1)+options.length)%options.length];
    drawSettings();$(`.panel-tabs [data-view="${state.panel}"]`)?.focus();
});

loadWorkspace().catch(error=>{ $('#workspace').innerHTML=blank(state);notice(error.message,true); });
