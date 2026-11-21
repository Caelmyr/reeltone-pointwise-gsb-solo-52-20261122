const entity = {'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'};
export const esc = value => String(value ?? '').replace(/[&<>"']/g, match => entity[match]);
export const $ = (selector, root = document) => root.querySelector(selector);
export const all = (selector, root = document) => [...root.querySelectorAll(selector)];
export const number = (form, name) => Number(form.elements.namedItem(name).value);
export const value = (form, name) => form.elements.namedItem(name).value;
export const checked = (form, name) => form.elements.namedItem(name).checked;
export const fmt = (n, digits = 3) => n === null || n === undefined ? '—' : Number(n).toLocaleString('zh-CN', {maximumFractionDigits:digits});
export const percent = (n, digits = 1) => n === null || n === undefined ? '—' : `${(Number(n) * 100).toFixed(digits)}%`;
export const span = rows => `${fmt(rows[0], 0)} 到 ${fmt(rows[1], 0)}`;
export const shorten = code => code ? `${code.slice(0, 12)}…` : '—';
export const scanURL = (scan, suffix = '') => `/api/scans/${scan}${suffix}`;
export const reelURL = (reel, suffix = '') => `/api/reels/${reel}${suffix}`;
export const dt = date => new Date(date).toLocaleString('zh-CN', {month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'});

export async function request(path, {method = 'GET', data, signal} = {}) {
    const options = {method, signal};
    if (data instanceof FormData) {
        options.body = data;
    } else if (data !== undefined) {
        options.headers = {'Content-Type':'application/json'};
        options.body = JSON.stringify(data);
    }
    const response = await fetch(path, options);
    let body;
    try { body = await response.json(); }
    catch { throw Error(`工作台返回了无法处理的响应 (${response.status})。`); }
    if (!response.ok) throw Error(body.error || `请求失败 (${response.status})。`);
    return body;
}
export const post = (path, data = {}) => request(path, {method:'POST', data});

export function boxInputs(prefix, arr, label = '') {
    return `${label ? `<small class="full">${esc(label)}</small>` : ''}<div class="box-row full">${['左列', '起始行', '右边界', '停止行'].map((name, i) => field(`${prefix}${i}`, name, arr[i], {type:'number',min:0,step:1})).join('')}</div>`;
}
export function readBox(form, prefix = 'box') {
    return [0, 1, 2, 3].map(i => number(form, `${prefix}${i}`));
}
export function field(name, label, initial, config = {}) {
    const {type = 'text',min,max,step,placeholder,maxlength,full,required = true,accept,readOnly} = config;
    const attrs = {min,max,step,placeholder,maxlength,accept};
    let extras = Object.entries(attrs).filter(([,val]) => val !== undefined).map(([key,val]) => `${key}="${esc(val)}"`).join(' ');
    if (required) extras += ' required';
    if (readOnly) extras += ' readonly';
    return `<label class="field${full ? ' full' : ''}">${esc(label)}<input name="${esc(name)}" type="${esc(type)}" ${type === 'file' ? '' : `value="${esc(initial)}"`} ${extras}></label>`;
}
export function select(name, label, options, selected, full = false) {
    return `<label class="field${full ? ' full' : ''}">${esc(label)}<select name="${esc(name)}">${options.map(([id,text]) => `<option value="${esc(id)}" ${String(selected) === String(id) ? 'selected' : ''}>${esc(text)}</option>`).join('')}</select></label>`;
}
export function check(name, label, active, full = false) {
    return `<label class="check${full ? ' full' : ''}"><input type="checkbox" name="${esc(name)}" ${active ? 'checked' : ''}> ${esc(label)}</label>`;
}
export function textarea(name, label, initial, length = 650) {
    return `<label class="field full">${esc(label)}<textarea name="${esc(name)}" maxlength="${length}">${esc(initial)}</textarea></label>`;
}
export const form = (id, fields, footer) => `<form id="${esc(id)}"><div class="fields">${fields}</div>${footer ? `<div class="form-footer">${footer}</div>` : ''}</form>`;
export const submit = text => `<button type="submit" class="strong">${esc(text)}</button>`;
export function action(id, text, data = {}, className = '') {
    return `<button type="button" ${className ? `class="${esc(className)}"` : ''} data-action="${esc(id)}" ${Object.entries(data).map(([key,val]) => `data-${key}="${esc(val)}"`).join(' ')}>${esc(text)}</button>`;
}
export function link(url, text, {download, target, css = ''} = {}) {
    return `<a href="${esc(url)}" class="link-button ${esc(css)}" ${download ? `download="${esc(download)}"` : ''} ${target ? 'target="_blank" rel="noopener"' : ''}>${esc(text)}</a>`;
}
export function metrics(rows) {
    return `<div class="metrics">${rows.map(([title, val]) => `<div class="metric"><small>${esc(title)}</small><b>${esc(val)}</b></div>`).join('')}</div>`;
}
export function jsonDetails(data, label = '查看读数明细') {
    return `<details><summary>${esc(label)}</summary><pre>${esc(JSON.stringify(data,null,2))}</pre></details>`;
}
export function record(title, caption, tools = '') {
    return `<div class="record"><div><strong>${esc(title)}</strong><small>${esc(caption)}</small></div><div class="record-tools">${tools}</div></div>`;
}
export function image(url, text, attributes = '') {
    return url ? `<img src="${esc(url)}" alt="${esc(text)}" ${attributes}>` : '';
}
export function listTable(head, rows) {
    return `<div class="table-wrap"><table class="data-table"><thead><tr>${head.map(t => `<th scope="col">${esc(t)}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${row.map(value => `<td>${esc(value)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}
export function statusClass(label) {
    return label === 'reviewed' || label === 'can_deliver' || label === 'checked' ? 'status-good' : label === 'hold' || label === 'partial' ? 'status-warning' : '';
}
export const TAKE = {draft:'草稿',reviewed:'已复核',hold:'暂缓'};
export const CUE = {excerpt:'摘段',splice:'接片',damage:'磨损',silence:'间歇',sync:'同步',leader:'引片'};
export const CUE_STATUS = {open:'待核对',checked:'已核对',omit:'略过'};
export const REVIEW = {omitted:'略过',can_deliver:'可交付',unreviewed:'待复核',partial:'仅覆盖部分',unread:'未监听'};

let noticeTimer;
export function notice(message, error = false) {
    const toast = $('#note');
    toast.textContent = message;
    toast.classList.toggle('error', error);
    toast.hidden = false;
    clearTimeout(noticeTimer);
    noticeTimer = setTimeout(() => { toast.hidden = true; }, error ? 9000 : 4700);
}
export function modal(title, body) {
    const frame = $('#modal');
    $('#modal-content').innerHTML = `<div class="modal-top"><h2 id="modal-heading">${esc(title)}</h2>${action('close-modal','关闭',{},'subtle')}</div><div class="modal-body">${body}</div>`;
    if (!frame.open) frame.showModal();
}
export function closeModal() {
    const frame = $('#modal');
    if (frame.open) frame.close();
}
