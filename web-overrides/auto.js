/* TrainItLocal automatic two-step workflow. Keeps manual mode untouched. */
(() => {
  const $ = (id) => document.getElementById(id);
  if (!$('autoPrompt') || !$('autoBuild')) return;
  const api = async (path, options = {}) => {
    const key = $('apiKey') && $('apiKey').value.trim();
    const headers = {'Content-Type':'application/json', ...(options.headers||{})};
    if (key) headers['X-API-Key'] = key;
    const r = await fetch(path, {...options, headers});
    const text = await r.text();
    let data; try { data = text ? JSON.parse(text) : {}; } catch { data = {detail:text}; }
    if (!r.ok) throw new Error(data.detail || data.error || `Request failed (${r.status})`);
    return data;
  };
  const status = (s) => { $('autoStatus').textContent = s; };
  const result = (s) => { $('autoResult').textContent = s; };
  $('autoBuild').addEventListener('click', async () => {
    const prompt = $('autoPrompt').value.trim();
    if (!prompt) { status('Describe the AI you want first.'); return; }
    $('autoBuild').disabled = true;
    $('autoDownload').hidden = true;
    try {
      status('Planning your AI…');
      const plan = await api('/api/auto/plan', {method:'POST', body:JSON.stringify({description:prompt})});
      result(JSON.stringify(plan.plan || plan, null, 2));
      status('Building dataset and training model…');
      const job = await api('/api/auto/build', {method:'POST', body:JSON.stringify({description:prompt, plan:plan.plan || plan})});
      const id = job.job_id;
      while (id) {
        const s = await api('/api/auto/status/' + encodeURIComponent(id));
        status(s.message || s.status || 'Working…');
        if (s.status === 'completed') {
          const d = s.result || {};
          $('autoDownload').href = d.download_url || '#';
          $('autoDownload').hidden = !d.download_url;
          $('autoDownload').onclick = async (event) => {
            if (!$('apiKey')?.value.trim()) return;
            event.preventDefault();
            const key = $('apiKey').value.trim();
            const response = await fetch(d.download_url, {headers:{'X-API-Key':key}});
            if (!response.ok) throw new Error('Could not download the trained package.');
            const blob = await response.blob(); const url = URL.createObjectURL(blob);
            const a = document.createElement('a'); a.href=url; a.download=`${d.experiment}-trained-model.zip`;
            document.body.appendChild(a); a.click(); a.remove(); setTimeout(()=>URL.revokeObjectURL(url),60000);
          };
          result(JSON.stringify(d, null, 2));
          status('Your deployable AI is ready.');
          break;
        }
        if (s.status === 'failed') throw new Error(s.error || 'Automatic build failed.');
        await new Promise(r => setTimeout(r, 1500));
      }
    } catch (e) { status('Build failed'); result(e.message); }
    finally { $('autoBuild').disabled = false; }
  });
})();
