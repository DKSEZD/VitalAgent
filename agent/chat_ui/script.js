document.addEventListener('DOMContentLoaded', () => {
    const $ = (id) => document.getElementById(id);
    const patientInput = $('patient-input');
    const segmentsInput = $('segments-input');
    const speedSelect = $('speed-select');
    const startBtn = $('start-btn');
    const pauseBtn = $('pause-btn');
    const resumeBtn = $('resume-btn');
    const nextBtn = $('next-btn');
    const stopBtn = $('stop-btn');
    const sessionNote = $('session-note');
    const alertFeed = $('alert-feed');
    const alertCount = $('alert-count');
    const canvas = $('ecg-canvas');
    const stripMode = $('strip-mode');
    const stripVitals = $('strip-vitals');
    const resumeLiveBtn = $('resume-live-btn');
    const chatWindow = $('chat-window');
    const messageInput = $('message-input');
    const sendBtn = $('send-btn');
    const clearBtn = $('clear-btn');

    let eventSource = null;
    let liveWindow = null;
    let displayWindow = null;
    let frozen = false;
    let alertTotal = 0;
    let chatBusy = false;
    let chatAbortController = null;
    let lastPaintAt = 0;
    const CHAT_TIMEOUT_MS = 90000;

    if (window.marked) window.marked.setOptions({ gfm: true, breaks: true });
    if (window.lucide) window.lucide.createIcons();

    startBtn.addEventListener('click', startSession);
    pauseBtn.addEventListener('click', () => controlSession('pause'));
    resumeBtn.addEventListener('click', () => controlSession('resume'));
    nextBtn.addEventListener('click', () => controlSession('next'));
    stopBtn.addEventListener('click', () => controlSession('stop'));
    speedSelect.addEventListener('change', () => {
        if (eventSource) controlSession('speed', { speed: Number(speedSelect.value) });
    });
    resumeLiveBtn.addEventListener('click', () => {
        frozen = false;
        displayWindow = liveWindow;
        stripMode.textContent = 'Latest analyzed window';
        resumeLiveBtn.classList.add('hidden');
    });

    document.querySelectorAll('.example-btn').forEach((button) => {
        button.addEventListener('click', () => sendMessage(button.textContent.trim()));
    });
    messageInput.addEventListener('input', () => {
        messageInput.style.height = 'auto';
        messageInput.style.height = `${messageInput.scrollHeight}px`;
        sendBtn.disabled = chatBusy || !messageInput.value.trim();
    });
    messageInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            if (!chatBusy && messageInput.value.trim()) sendMessage(messageInput.value);
        }
    });
    sendBtn.addEventListener('click', () => {
        if (!chatBusy) sendMessage(messageInput.value);
    });
    clearBtn.addEventListener('click', async () => {
        try {
            await fetchJson('/api/session/reset_chat', {});
            chatWindow.querySelectorAll('.message:not(.welcome-msg)').forEach((node) => node.remove());
        } catch (error) {
            sessionNote.textContent = `Clear failed: ${error.message}`;
        }
    });

    async function startSession() {
        const segments = segmentsInput.value.split(',').map((item) => item.trim()).filter(Boolean);
        startBtn.disabled = true;
        sessionNote.textContent = 'Starting local ECG replay…';
        closeEventSource();
        try {
            const response = await fetchJson('/api/session/start', {
                patient: patientInput.value.trim(),
                segments,
                speed: Number(speedSelect.value),
                window_seconds: 10,
            });
            alertFeed.replaceChildren(makeEmptyState('No alerts yet.'));
            alertTotal = 0;
            alertCount.textContent = '0';
            frozen = false;
            liveWindow = null;
            displayWindow = null;
            applySnapshot(response.snapshot);
            sessionNote.textContent = 'Connected · receiving analyzed windows';
            openEventSource();
        } catch (error) {
            sessionNote.textContent = `Start failed: ${error.message}`;
        } finally {
            startBtn.disabled = false;
        }
    }

    function openEventSource() {
        closeEventSource();
        eventSource = new EventSource('/api/session/events');
        eventSource.onmessage = (message) => {
            const event = JSON.parse(message.data);
            if (event.type === 'window') handleWindow(event);
            else if (event.type === 'alert') handleAlert(event.alert);
            else if (event.type === 'status') applySnapshot(event.snapshot);
            else if (event.type === 'done') {
                applySnapshot(event.snapshot);
                sessionNote.textContent = event.snapshot.error
                    ? `Session ended: ${event.snapshot.error}`
                    : 'Monitoring session complete';
                closeEventSource();
            } else if (event.type === 'error') {
                sessionNote.textContent = `Stream error: ${event.message}`;
            }
        };
        eventSource.onerror = () => {
            if (eventSource) sessionNote.textContent = 'Event stream reconnecting…';
        };
    }

    function closeEventSource() {
        if (eventSource) eventSource.close();
        eventSource = null;
    }

    async function controlSession(action, body = {}) {
        try {
            const response = await fetchJson(`/api/session/${action}`, body);
            applySnapshot(response.snapshot);
        } catch (error) {
            sessionNote.textContent = error.message;
        }
    }

    function handleWindow(event) {
        liveWindow = event;
        if (!frozen) displayWindow = event;
        const hr = Number.isFinite(event.hr_bpm) ? `${event.hr_bpm.toFixed(0)} bpm` : '—';
        stripVitals.textContent = `HR ${hr} · Rhythm ${event.rhythm_class || '—'}`;
    }

    function handleAlert(alert) {
        alertTotal += 1;
        alertCount.textContent = String(alertTotal);
        const empty = alertFeed.querySelector('.empty-state');
        if (empty) empty.remove();
        alertFeed.prepend(createAlertCard(alert));
        if (alert.window) {
            displayWindow = {
                signal_preview: alert.window.signal_preview,
                fs_preview: alert.window.fs_preview,
                offset_s: alert.window.offset,
            };
        }
        frozen = true;
        stripMode.textContent = `Frozen at alert · ${formatTime(alert.offset_s)}`;
        resumeLiveBtn.classList.remove('hidden');
    }

    function createAlertCard(alert) {
        const card = document.createElement('button');
        card.type = 'button';
        card.className = `alert-card urgency-${String(alert.urgency || 'none').toLowerCase()}`;
        const top = document.createElement('div');
        top.className = 'alert-card-top';
        const time = document.createElement('span');
        time.textContent = formatTime(alert.offset_s);
        const urgency = document.createElement('span');
        urgency.textContent = String(alert.urgency || 'none').toUpperCase();
        top.append(time, urgency);
        const rules = document.createElement('div');
        rules.className = 'alert-rules';
        rules.textContent = (alert.triggered_rules || []).join(' · ') || 'contextual judge';
        const reason = document.createElement('p');
        reason.textContent = alert.reason || 'Monitoring rule triggered.';
        card.append(top, rules, reason);
        card.addEventListener('click', () => sendMessage('Why did I just receive an alert?'));
        return card;
    }

    function applySnapshot(snapshot) {
        if (!snapshot) return;
        const totalKnown = Number.isFinite(snapshot.duration_s) && snapshot.duration_s > 0;
        const completed = Boolean(snapshot.completed);
        const position = completed && totalKnown ? snapshot.duration_s : Number(snapshot.offset_s_global || 0);
        const percent = totalKnown ? Math.max(0, Math.min(100, completed ? 100 : 100 * position / snapshot.duration_s)) : null;
        $('status-time').textContent = `${formatTime(position)} / ${totalKnown ? formatTime(snapshot.duration_s) : '--:--:--'}`;
        $('status-percent').textContent = percent === null ? '?%' : `${percent.toFixed(1)}%`;
        $('status-speed').textContent = `${Number(snapshot.speed || 0).toLocaleString()}×`;
        $('status-state').textContent = snapshot.error ? 'ERROR' : completed ? 'DONE' : snapshot.paused ? 'PAUSED' : snapshot.fast_forwarding ? 'FF' : snapshot.running ? 'RUNNING' : 'IDLE';
        $('status-segment').textContent = `seg ${snapshot.current_segment_id || '—'}`;
        $('progress-fill').style.width = `${percent || 0}%`;
    }

    function setBusy(busy) {
        chatBusy = busy;
        sendBtn.disabled = busy || !messageInput.value.trim();
        document.querySelectorAll('.example-btn').forEach((b) => { b.disabled = busy; });
        if (alertFeed) alertFeed.classList.toggle('chat-busy', busy);
    }

    async function sendMessage(rawText) {
        const text = String(rawText || '').trim();
        if (!text || chatBusy) return;
        appendUserMessage(text);
        messageInput.value = '';
        messageInput.style.height = 'auto';
        setBusy(true);
        chatAbortController = new AbortController();
        const assistant = appendAssistantMessage();
        const logs = [];
        let timedOut = false;
        let streamSettled = false;
        const settleStream = () => {
            if (streamSettled) return;
            streamSettled = true;
            clearTimeout(timeoutId);
        };
        const timeoutId = setTimeout(() => {
            timedOut = true;
            chatAbortController?.abort();
        }, CHAT_TIMEOUT_MS);
        try {
            const response = await fetch('/api/session/chat_stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text }),
                signal: chatAbortController.signal,
            });
            if (!response.ok) {
                const errorPayload = await response.json().catch(() => ({}));
                throw new Error(errorPayload.error || `HTTP ${response.status}`);
            }
            await consumeSSE(response, (event) => {
                if (event.type === 'status') {
                    logs.push(event.message);
                    renderAssistant(assistant, '', logs, null);
                } else if (event.type === 'answer') {
                    renderAssistant(assistant, event.answer, logs, null);
                } else if (event.type === 'final') {
                    settleStream();
                    renderAssistant(assistant, event.answer, logs, event.tool_results || []);
                } else if (event.type === 'error') {
                    settleStream();
                    throw new Error(event.message);
                } else if (event.type === 'done') {
                    settleStream();
                }
            });
        } catch (error) {
            if (timedOut) renderAssistant(assistant, 'Request timed out — try again.', logs, null);
            else if (error.name !== 'AbortError') renderAssistant(assistant, `Request failed: ${error.message}`, logs, null);
        } finally {
            clearTimeout(timeoutId);
            chatAbortController = null;
            setBusy(false);
        }
    }

    async function consumeSSE(response, onEvent) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        try {
            let finished = false;
            while (!finished) {
                const { value, done } = await reader.read();
                buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
                const blocks = buffer.split(/\r?\n\r?\n/);
                buffer = blocks.pop() || '';
                for (const block of blocks) {
                    const data = block.split(/\r?\n/).filter((line) => line.startsWith('data:')).map((line) => line.slice(5).trim()).join('\n');
                    if (!data) continue;
                    const event = JSON.parse(data);
                    onEvent(event);
                    // Terminate on the explicit done event instead of waiting for the
                    // server to close the stream; keep-alive may hold the socket open.
                    if (event.type === 'done') { finished = true; break; }
                }
                if (done) break;
            }
        } finally {
            try { await reader.cancel(); } catch (e) { /* stream already closed */ }
        }
    }

    function appendUserMessage(text) {
        const node = document.createElement('div');
        node.className = 'message user-msg';
        const content = document.createElement('div');
        content.className = 'msg-content';
        content.textContent = text;
        node.appendChild(content);
        chatWindow.appendChild(node);
        scrollChat(true);
    }

    function appendAssistantMessage() {
        const node = document.createElement('div');
        node.className = 'message assistant-msg';
        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.textContent = 'AI';
        const content = document.createElement('div');
        content.className = 'msg-content';
        content.textContent = 'Planning…';
        node.append(avatar, content);
        chatWindow.appendChild(node);
        scrollChat(true);
        return content;
    }

    function renderAssistant(container, answer, logs, toolResults) {
        const stick = isNearBottom();
        container.replaceChildren();
        if (logs.length) {
            const details = document.createElement('details');
            details.className = 'logs-details';
            const summary = document.createElement('summary');
            summary.textContent = 'Processing steps';
            details.appendChild(summary);
            logs.forEach((log) => {
                const row = document.createElement('div');
                row.className = 'log-row';
                row.textContent = log;
                details.appendChild(row);
            });
            container.appendChild(details);
        }
        const answerNode = document.createElement('div');
        answerNode.className = 'markdown-body';
        answerNode.innerHTML = renderMarkdown(answer || (logs.length ? 'Thinking…' : ''));
        container.appendChild(answerNode);
        if (Array.isArray(toolResults) && toolResults.length) container.appendChild(createEvidence(toolResults));
        scrollChat(stick);
    }

    function createEvidence(toolResults) {
        const details = document.createElement('details');
        details.className = 'evidence-details';
        const summary = document.createElement('summary');
        summary.textContent = `Evidence · ${toolResults.length} tool result${toolResults.length === 1 ? '' : 's'}`;
        details.appendChild(summary);
        toolResults.forEach((result) => {
            const item = document.createElement('div');
            item.className = 'evidence-item';
            const title = document.createElement('strong');
            title.textContent = `${result.tool_name || 'tool'} · ${result.success ? 'success' : 'failed'}`;
            const pre = document.createElement('pre');
            const hasData = result.data && Object.keys(result.data).length > 0;
            const body = result.success
                ? result.data
                : (result.error || (hasData ? result.data : 'No data returned.'));
            pre.textContent = typeof body === 'string' ? body : JSON.stringify(body, null, 2);
            item.append(title, pre);
            details.appendChild(item);
        });
        return details;
    }

    function renderMarkdown(text) {
        const raw = String(text || '');
        if (!window.marked || !window.DOMPurify) return escapeHTML(raw).replace(/\n/g, '<br>');
        return window.DOMPurify.sanitize(window.marked.parse(raw));
    }

    function makeEmptyState(text) {
        const node = document.createElement('p');
        node.className = 'empty-state';
        node.textContent = text;
        return node;
    }

    async function fetchJson(url, body) {
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body || {}),
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
        return payload;
    }

    function formatTime(value) {
        const total = Math.max(0, Math.round(Number(value) || 0));
        const hours = Math.floor(total / 3600);
        const minutes = Math.floor((total % 3600) / 60);
        const seconds = total % 60;
        return [hours, minutes, seconds].map((part) => String(part).padStart(2, '0')).join(':');
    }

    function isNearBottom() { return chatWindow.scrollHeight - chatWindow.scrollTop - chatWindow.clientHeight < 80; }
    function scrollChat(force) { if (force || isNearBottom()) chatWindow.scrollTop = chatWindow.scrollHeight; }
    function escapeHTML(value) { return String(value).replace(/[&<>'"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[char])); }

    function paintLoop(timestamp) {
        if (timestamp - lastPaintAt >= 333) {
            drawECG(displayWindow && displayWindow.signal_preview);
            lastPaintAt = timestamp;
        }
        requestAnimationFrame(paintLoop);
    }

    function drawECG(signal) {
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const width = Math.max(1, Math.floor(rect.width * dpr));
        const height = Math.max(1, Math.floor(rect.height * dpr));
        if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, rect.width, rect.height);
        ctx.strokeStyle = 'rgba(59,130,246,.12)';
        ctx.lineWidth = 1;
        for (let x = 0; x < rect.width; x += 20) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, rect.height); ctx.stroke(); }
        for (let y = 0; y < rect.height; y += 20) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(rect.width, y); ctx.stroke(); }
        if (!Array.isArray(signal) || !signal.length) return;
        const finite = signal.filter(Number.isFinite);
        if (!finite.length) return;
        const min = Math.min(...finite); const max = Math.max(...finite); const span = Math.max(max - min, 1e-9);
        ctx.strokeStyle = frozen ? '#fbbf24' : '#60a5fa';
        ctx.lineWidth = 1.7;
        ctx.beginPath();
        let drawing = false;
        signal.forEach((value, index) => {
            if (!Number.isFinite(value)) { drawing = false; return; }
            const x = signal.length === 1 ? 0 : index * rect.width / (signal.length - 1);
            const y = span <= 1e-8 ? rect.height / 2 : rect.height - 14 - ((value - min) / span) * (rect.height - 28);
            if (!drawing) { ctx.moveTo(x, y); drawing = true; } else ctx.lineTo(x, y);
        });
        ctx.stroke();
    }

    requestAnimationFrame(paintLoop);
});
