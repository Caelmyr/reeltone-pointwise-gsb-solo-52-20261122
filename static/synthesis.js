import {$, action, check, esc, field, form, image, jsonDetails, link, metrics, modal, notice, post, select, submit} from './ui.js';

const DEFAULTS = {
    row_rate: 7200, aperture_width: 96, slit_rows: 1, margin_fraction: 0.03, knee: 0.85,
    grain_sigma: 0.006, base: 0.135, level: 0.86, polarity: 'clear',
    title: 'WAV 合成测试声轨',
};

export function synthDefaults() {
    return {...DEFAULTS};
}

export function synthModal(state) {
    state.synth = state.synth || synthDefaults();
    const d = state.synth;
    const number = (name, label, min, max, step, full = false) =>
        field(name, label, d[name], {type: 'number', min, max, step, full});
    const fields = [
        field('wav', '单声道测试音 WAV（16 位 PCM，≤ 20 秒）', '',
            {type: 'file', accept: '.wav,audio/wav,audio/x-wav', full: true}),
        field('title', '测试图名称', d.title, {full: true, maxlength: 90}),
        number('row_rate', '行速（行/秒）', 500, 48000, 1),
        number('aperture_width', '读窗宽度（列）', 24, 600, 1),
        number('slit_rows', '读窗高度（行）', 1, 5, 1),
        number('margin_fraction', '边缘保留比例', 0.005, 0.15, 0.001),
        number('knee', '软限幅拐点（满幅比例）', 0.4, 0.995, 0.005),
        number('grain_sigma', '颗粒噪声标准差（0 关闭）', 0, 0.06, 0.001),
        number('base', '片基透射', 0, 0.45, 0.005),
        number('level', '满幅透射', 0.55, 1, 0.005),
        select('polarity', '轨道极性', [['clear', '透光为信号（正片）'], ['dark', '遮光为信号（反相）']], d.polarity),
    ].join('');
    modal('从 WAV 生成合成光学声轨测试图',
        `<p class="panel-note status-warning">输出为<b>测试用合成图</b>，不是真实胶片扫描。图内首部带会烧入中英文警示标注。
        重采样用抗混叠低通：行速读不到的高音会被滤除，不会折叠成新的低音；峰值经软限幅且读窗始终留边，响峰不会挤满整条。</p>
        ${form('synth-form', fields, `${submit('生成预览与回读对照')}${action('close-modal', '取消', {}, 'subtle')}`)}
        <div id="synth-result"></div>`);
}

function readSynthForm(node) {
    const get = name => Number(node.elements.namedItem(name).value);
    return {
        row_rate: get('row_rate'), aperture_width: get('aperture_width'), slit_rows: get('slit_rows'),
        margin_fraction: get('margin_fraction'), knee: get('knee'), grain_sigma: get('grain_sigma'),
        base: get('base'), level: get('level'),
        polarity: node.elements.namedItem('polarity').value,
        title: node.elements.namedItem('title').value,
    };
}

function bandTable(report) {
    const rows = report.readback.band_table.map(row =>
        `<tr><td>${row.band_hz[0]}–${row.band_hz[1]} Hz</td><td>${fmtDb(row.reference_db)}</td><td>${fmtDb(row.recovered_db)}</td></tr>`).join('');
    return `<div class="table-wrap"><table class="data-table"><thead><tr><th scope="col">频段</th><th scope="col">限带参考 dB</th><th scope="col">实际回读 dB</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

function fmtDb(value) {
    return value === null ? '—' : `${value.toLocaleString('zh-CN', {maximumFractionDigits: 2})} dB`;
}

function metricCards(report) {
    const rb = report.readback, lc = report.row_clock, lim = report.limiter, dr = report.dynamic_range, q = report.quiet_retention;
    const fold = lc.foldback.above_nyquist_present ? `${lc.foldback.foldback_suppression_db} dB 抑制` : '无超限分量';
    return metrics([
        ['回读相关系数', rb.correlation.toFixed(5)],
        ['回读 RMS 误差', fmtDb(rb.rms_error_dbfs)],
        ['最大样本误差', rb.max_abs_error.toFixed(4)],
        ['滤除高频能量占比', `${lc.removed_above_nyquist_percent}%`],
        ['新奈奎斯特频率', `${lc.nyquist_hz} Hz`],
        ['高音折叠抑制', fold],
        ['软限幅行数', `${lim.rows_limited} / ${lc.rows}`],
        ['单峰最大压缩', fmtDb(lim.strongest_row_reduction_db)],
        ['图案有效动态范围', dr.effective_dynamic_range_db === null ? '不可测（近乎静音）' : `${dr.effective_dynamic_range_db} dB`],
        ['信号区触边行', rb.touch_edge_rows_in_signal],
        ['读窗高度理论高频衰减', report.pattern.slit_rows > 1 ? `${report.pattern.slit_high_octave_attenuation_db.theoretical_db_at_0_75_nyquist} dB` : '0 dB（单行）'],
        ['混叠代理电平', fmtDb(rb.alias_proxy_rms_dbfs)],
        ['弱音段占比（RMS≤0.1）', `${(q.quiet_window_fraction * 100).toFixed(1)}%`],
        ['弱音段回读误差', fmtDb(q.quiet_section_error_dbfs)],
    ]);
}

function levelStats(report) {
    const src = report.source.statistics;
    const ref = report.band_limited_reference.statistics;
    const got = report.readback.statistics;
    const row = (label, s) => `<tr><td>${label}</td><td>${s.peak.toFixed(4)}</td><td>${fmtDb(s.peak_dbfs)}</td><td>${s.rms.toFixed(4)}</td><td>${fmtDb(s.rms_dbfs)}</td></tr>`;
    return `<div class="table-wrap"><table class="data-table"><thead><tr><th scope="col">信号</th><th scope="col">峰值</th><th scope="col">峰值 dBFS</th><th scope="col">RMS</th><th scope="col">RMS dBFS</th></tr></thead>
        <tbody>${row('原始 WAV', src)}${row('限带参考（重采样+限幅后）', ref)}${row('图案实际回读', got)}</tbody></table></div>`;
}

export function renderSynthResult(body) {
    const report = body.report;
    const token = body.job;
    const figures = body.figures;
    const html = `
        <div class="synth-banner">● 测试用合成图 · SYNTHETIC TEST STRIP · 非真实胶片扫描</div>
        ${metricCards(report)}
        <h3>原音 → 限带参考 → 实际回读</h3>
        ${levelStats(report)}
        <h3>频段对照（参考 vs 回读）</h3>
        ${bandTable(report)}
        <h3>对照图</h3>
        <div class="preview-grid">
            <div class="preview-card">${image(figures.comparison, '波形对照')}<small>三层波形：原音 / 限带参考 / 实际回读</small></div>
            <div class="preview-card">${image(figures.spectrum, '频谱对照')}<small>频谱：抗混叠后参考与回读，红线为奈奎斯特</small></div>
            <div class="preview-card">${image(figures.strip, '合成测试全图')}<small>16 位灰度测试图，青色为读窗，琥珀色为信号行范围</small></div>
        </div>
        <h3>为装进规格付出的代价（说明）</h3>
        <ul class="synth-notes">${report.explanation.map(line => `<li>${esc(line)}</li>`).join('')}</ul>
        <div class="panel-note">高频去除：高于 ${report.row_clock.nyquist_hz} Hz 的能量占原音 ${report.row_clock.removed_above_nyquist_percent}%，
        由行速 ${report.row_clock.row_rate} 行/秒的抗混叠滤波移除，未产生可闻折叠分量（混叠代理 ${fmtDb(report.readback.alias_proxy_rms_dbfs)}${
            report.row_clock.foldback.above_nyquist_present ? `，折叠抑制 ${report.row_clock.foldback.foldback_suppression_db} dB` : '，源内无超限频率'}）。
        动态范围：读窗 ${report.pattern.width_columns} 列给出约 ${report.dynamic_range.aperture_quantization_levels} 级边缘位置，
        有效动态范围 ${report.dynamic_range.effective_dynamic_range_db === null ? report.dynamic_range.dynamic_range_note : `${report.dynamic_range.effective_dynamic_range_db} dB`}，残余底噪 ${fmtDb(report.quiet_retention.residual_equivalent_dbfs)}。
        峰值控制：${report.limiter.rows_limited} 行超过拐点 ${report.limiter.knee_fraction}，单峰最多压缩 ${fmtDb(report.limiter.strongest_row_reduction_db)}，
        图案开口限制在 ${report.limiter.openness_min}–${report.limiter.openness_max}，两侧各留 ${report.pattern.edge_margin_fraction * 100}% 边缘。</div>
        <div class="form-footer">
            ${action('synth-commit', '确认并保存到工作库', {job: token}, 'strong')}
            ${link(`/api/synthesis/${token}/strip.png`, '下载测试全图 PNG', {download: 'reeltone-synthetic-strip.png'})}
            ${link(`/api/synthesis/${token}/settings.json`, '下载可重开的读片设置 JSON', {download: 'synthetic-settings.json'})}
        </div>
        ${jsonDetails(report, '查看完整读数 JSON')}`;
    $('#synth-result').innerHTML = html;
}

export function connectSynthesis(state) {
    return {
        async preview(node) {
            if (!node.reportValidity()) return;
            const file = node.elements.namedItem('wav').files[0];
            if (!file) throw Error('请先选择单声道 WAV 测试音。');
            state.synth = readSynthForm(node);
            const data = new FormData();
            data.append('wav', file);
            data.append('settings', JSON.stringify(state.synth));
            const body = await post('/api/synthesis/preview', data);
            state.synthJob = body.job;
            renderSynthResult(body);
            notice('合成预览与回读对照已生成。确认后可保存到工作库。');
        },
        async commit(token) {
            const answer = await post('/api/synthesis/commit', {job: token, reel_title: `合成测试卷 · ${state.synth.title}`});
            notice('已保存合成测试图，正在用正常读片台打开。');
            return answer;
        },
    };
}
