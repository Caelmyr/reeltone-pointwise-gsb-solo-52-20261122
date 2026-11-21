import {action, esc, field, form, image, jsonDetails, metrics, modal, post, reelURL, request, select, span, submit} from './ui.js';

export function joinWizard(state) {
    const scans = state.reel.scans.filter(scan => scan.metadata.width === state.reel.scans[0]?.metadata.width);
    if (scans.length < 2) {
        return '<p class="panel-note">本卷至少需要两张同宽扫描才能接续。</p>';
    }
    const head = state.joinWizard?.head || scans[0].id;
    const tail = state.joinWizard?.tail || scans[1].id;
    const options = scans.map(scan => [scan.id, `${scan.title}（${scan.metadata.height} 行）`]);
    return form('join-form',
        select('head_scan', '片头扫描（前段）', options, head, true) +
        select('tail_scan', '片尾扫描（后段）', options, tail, true) +
        rowMarkFields(state, 'head') + rowMarkFields(state, 'tail') +
        select('mode', '接缝方式', [['quality', '按清晰度选源（瑕疵处换清楚一侧）'], ['single', '重叠中点单一切缝']],
               state.joinWizard?.mode || 'quality', true) +
        field('crossfade', '切缝过渡行数', state.joinWizard?.crossfade ?? 6, {type:'number', min:0, max:40, step:1, full:true}) +
        field('min_segment', '最短来源段（行）', state.joinWizard?.min_segment ?? 48, {type:'number', min:4, max:4000, step:1, full:true}),
        `${submit('查找重叠候选')}${action('join-use-selection-head', '采用片头当前选框')}${action('join-use-selection-tail', '采用片尾当前选框')}`);
}

function rowMarkFields(state, side) {
    const mark = state.joinWizard?.marks?.[side] || ['', ''];
    return `<label class="field full">${side === 'head' ? '片头图上的大概重叠行（可留空）' : '片尾图上的大概重叠行（可留空）'}
        <div class="box-row">
        <input name="${side}_row0" type="number" min="0" step="1" value="${mark[0]}" placeholder="起始行">
        <input name="${side}_row1" type="number" min="1" step="1" value="${mark[1]}" placeholder="停止行">
        </div></label>`;
}

export function readJoinForm(node) {
    const value = name => node.elements.namedItem(name).value;
    const mark = side => {
        const a = value(`${side}_row0`), b = value(`${side}_row1`);
        return a !== '' && b !== '' ? [Number(a), Number(b)] : null;
    };
    return {
        head_scan: value('head_scan'), tail_scan: value('tail_scan'),
        head_rows: mark('head'), tail_rows: mark('tail'),
        mode: value('mode'), crossfade: Number(value('crossfade')),
        min_segment: Number(value('min_segment')),
    };
}

export async function openJoinWizard(state) {
    state.joinWizard = state.joinWizard || {marks: {}};
    modal('两段扫描纵向接续', `<p class="panel-note">两次扫描宽度与走片方向一致、只有纵向归位差异时使用。先在两张图上各框一个大概的重叠段，工作台会给出多个对齐候选并等你确认；确认后保存一张可继续监听的连续边带，原两张扫描保持不动。</p>
        <div id="join-step">${joinWizard(state)}</div>`);
}

function candidateRows(candidates) {
    return candidates.map((candidate, index) => `<label class="check full join-candidate">
        <input type="radio" name="offset" value="${candidate.offset}" ${index === 0 ? 'checked' : ''}>
        <span><b>候选 ${index + 1} · 偏移 ${candidate.offset >= 0 ? '+' : ''}${candidate.offset} 行</b>
        <small>相关 ${fmt3(candidate.score)} · 加权 ${fmt3(candidate.weighted_score)} · 共同行 ${candidate.overlap_rows} · 成片 ${candidate.accepted_rows} 行</small></span>
    </label>`).join('');
}

function fmt3(value) { return Number(value).toFixed(4); }

export async function runCandidateSearch(state, node) {
    const reel = state.reel.reel.id;
    const payload = readJoinForm(node);
    state.joinWizard = {...state.joinWizard, ...payload, marks: {head: [payload.head_rows?.[0] ?? '', payload.head_rows?.[1] ?? ''],
                                                                   tail: [payload.tail_rows?.[0] ?? '', payload.tail_rows?.[1] ?? '']}};
    const result = await post(reelURL(reel, '/joins/candidates'), payload);
    state.joinWizard.candidates = result;
    const body = `<div class="box">${metrics([['首选偏移', `${result.candidates[0].offset >= 0 ? '+' : ''}${result.candidates[0].offset} 行`],
        ['首选相关', fmt3(result.candidates[0].score)], ['共同行', result.candidates[0].overlap_rows]])}
        ${result.ambiguous ? `<p class="panel-note status-warning">${esc(result.notice)}</p>` : `<p class="panel-note">${esc(result.notice)}</p>`}
        ${result.curve_image ? image(result.curve_image, '纵向偏移相关曲线') : ''}
        <div class="join-candidates">${candidateRows(result.candidates)}</div></div>
        <div class="form-footer">${action('join-back', '返回调整范围')}${submit('生成接续方案预览')}</div>`;
    modal('选择对齐候选', form('join-candidate-form', body, ''));
}

export async function runPlanPreview(state, node) {
    const reel = state.reel.reel.id;
    const offset = Number(node.elements.namedItem('offset').value);
    const payload = {...basePayload(state), offset};
    const result = await post(reelURL(reel, '/joins/preview'), payload);
    state.joinWizard.preview = result;
    state.joinWizard.chosenOffset = offset;
    const manifest = result.manifest;
    const segmentRows = manifest.segments.map(segment => {
        const scan = segment.source === 'head' ? manifest.head_title : manifest.tail_title;
        return `<tr><td>${segment.source === 'head' ? '片头' : '片尾'}</td><td>${span([segment.output_start, segment.output_stop])}</td>
        <td>${esc(scan)}</td><td>${span([segment.source_first_row, segment.source_after_last_row])}</td><td>${segment.rows}</td><td>${esc(segment.reason)}</td></tr>`;
    }).join('');
    const body = `${image(result.preview, '接续方案预览（颜色区分来源段）')}
        ${metrics([['对齐偏移', `${manifest.offset_rows >= 0 ? '+' : ''}${manifest.offset_rows} 行`], ['重叠行', manifest.overlap.rows],
            ['来源切换', manifest.clarity.switch_count], ['过渡行', manifest.crossfade_rows], ['来源段数', result.segment_count]])}
        <div class="table-wrap"><table class="data-table"><thead><tr><th>来源</th><th>成片行</th><th>原扫描</th><th>原片行</th><th>行数</th><th>选由</th></tr></thead>
        <tbody>${segmentRows}</tbody></table></div>
        <p class="panel-note">${esc(manifest.warranty)}</p>
        <div class="form-footer">${action('join-back-candidates', '返回选候选')}${field('title', '连续边带名称', state.joinWizard.title || '接续连续边带', {full:true, maxlength:110})}
        ${submit('确认并保存连续边带')}</div>${jsonDetails(manifest, '查看完整来源与读数')}`;
    modal('核对接续方案', form('join-confirm-form', body, ''));
}

export async function commitJoin(state, node) {
    const reel = state.reel.reel.id;
    const payload = {...basePayload(state), offset: state.joinWizard.chosenOffset,
                     title: node.elements.namedItem('title').value};
    const result = await post(reelURL(reel, '/joins'), payload);
    state.reel = await request(reelURL(reel));
    state.library = await request('/api/library');
    modal('连续边带已保存', `<p class="panel-note status-good">原始两张扫描仍在卷内；新边带标记为「接续生成」，可用原读片台继续监听，来源段可随时查询。</p>
        ${image(result.preview, '已保存的接续方案')}
        <div class="form-footer">${action('join-open-output', '打开连续边带', {id: result.join.output_scan_id}, 'strong')}
        ${action('join-view-manifest', '查看来源明细', {id: result.join.id})}${action('close-modal', '关闭')}</div>`);
}

function basePayload(state) {
    const wizard = state.joinWizard;
    return {head_scan: wizard.head_scan, tail_scan: wizard.tail_scan, head_rows: wizard.head_rows, tail_rows: wizard.tail_rows,
            mode: wizard.mode, crossfade: wizard.crossfade, min_segment: wizard.min_segment};
}
