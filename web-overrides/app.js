const $ = id => document.getElementById(id);

const chatState = { messages: [] };

async function api(path, options={}) {
  const key = $('apiKey') && $('apiKey').value.trim();
  const headers = {...(options.headers || {})};
  if(key) headers['X-API-Key'] = key;
  options.headers = headers;
  const r = await fetch(path, options);
  const text = await r.text();
  let data;
  try { data = JSON.parse(text); } catch { data = {raw:text}; }
  if (!r.ok) throw new Error(data.detail || data.raw || `HTTP ${r.status}`);
  return data;
}

async function checkHealth(){
  try {
    await api('/health');
    $('health').textContent = 'API online';
  } catch(e) {
    $('health').textContent = 'API error';
  }
}
checkHealth();

async function uploadDataset(){
  const file = $('dataset').files[0];
  if(!file) return alert('Choose a dataset first.');
  const fd = new FormData();
  fd.append('file', file);
  try {
    const data = await api('/api/datasets/upload',{method:'POST',body:fd});
    $('uploadResult').textContent = JSON.stringify(data,null,2);
    window.datasetId = data.dataset_id;
    const analysis = await api('/api/datasets/analyze?dataset_id='+encodeURIComponent(data.dataset_id),{method:'POST'});
    $('uploadResult').textContent += "\n\nANALYSIS\n"+JSON.stringify(analysis.analysis,null,2);
    $('planResult').textContent = 'Dataset ready. Describe your training goal above.';
  } catch(e) { $('uploadResult').textContent = e.message; }
}

async function makePlan(){
  const prompt = $('prompt').value.trim();
  if(!prompt) return alert('Describe the training task first.');
  try {
    if(!window.datasetId) return alert('Upload a dataset first.');
    const data = await api('/api/planner/plan?dataset_id='+encodeURIComponent(window.datasetId),{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({prompt})
    });
    $('planResult').textContent = JSON.stringify(data.plan,null,2);
    $('config').value = JSON.stringify(data.plan,null,2);
  } catch(e) { $('planResult').textContent = e.message; }
}

async function startTraining(){
  let config;
  try { config = JSON.parse($('config').value); }
  catch { return alert('Training config must be valid JSON.'); }
  const modelName = $('modelName').value.trim();
  if(modelName) config.experiment_name = modelName;
  if(!config.experiment_name) return alert('Give the model a name before training.');
  $('config').value = JSON.stringify(config,null,2);
  try {
    const data = await api('/api/training/start',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({config})
    });
    $('jobBox').classList.remove('hidden');
    $('jobId').textContent = data.job_id;
    pollJob(data.job_id);
  } catch(e) { alert(e.message); }
}

async function pollJob(id){
  try {
    const data = await api('/api/training/'+encodeURIComponent(id));
    $('jobStatus').textContent = data.status;
    $('jobResult').textContent = JSON.stringify(data,null,2);
    if(data.status === 'queued' || data.status === 'running'){
      setTimeout(()=>pollJob(id),1000);
    } else if(data.status === 'completed'){
      const rec = data.result && data.result.experiment_record;
      const name = rec && rec.config && rec.config.experiment_name;
      if(name){
        $('modelName').value = name;
        $('predictExperiment').value = name;
        $('generateExperiment').value = name;
        $('chatExperiment').value = name;
        $('downloadExperiment').value = name;
      }
    }
  } catch(e) { $('jobResult').textContent = e.message; }
}

async function runGeneration(){
  const experiment = $('generateExperiment').value.trim();
  const prompt = $('generatePrompt').value;
  if(!experiment || !prompt) return alert('Enter an experiment name and prompt.');
  try {
    const data = await api('/api/generate', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({experiment, prompt})
    });
    $('generateResult').textContent = data.text;
  } catch(e) { $('generateResult').textContent = e.message; }
}

async function runPrediction(){
  const experiment = $('predictExperiment').value.trim();
  if(!experiment) return alert('Enter the experiment name to predict with.');
  let records;
  try { records = JSON.parse($('predictRecords').value); }
  catch { return alert('Records must be valid JSON (an array of objects).'); }
  if(!Array.isArray(records)) return alert('Records must be a JSON array of objects.');
  try {
    const data = await api('/api/predict',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({experiment, records})
    });
    $('predictResult').textContent = JSON.stringify(data.predictions,null,2);
  } catch(e) { $('predictResult').textContent = e.message; }
}

function renderChat(){
  const box = $('chatMessages');
  if(!box) return;
  box.innerHTML = '';
  for(const msg of chatState.messages){
    const item = document.createElement('div');
    item.className = `chat-message ${msg.role}`;
    const label = document.createElement('strong');
    label.textContent = msg.role === 'user' ? 'You' : 'Model';
    const text = document.createElement('div');
    text.textContent = msg.content;
    item.append(label, text);
    box.appendChild(item);
  }
  box.scrollTop = box.scrollHeight;
}

function clearChat(){
  chatState.messages = [];
  renderChat();
  $('chatResult').textContent = 'Conversation cleared.';
}

function handleChatKey(event){
  if(event.key === 'Enter' && !event.shiftKey){
    event.preventDefault();
    sendChat();
  }
}

async function sendChat(){
  const experiment = $('chatExperiment').value.trim();
  const input = $('chatInput').value.trim();
  if(!experiment) return alert('Enter the language-model experiment name.');
  if(!input) return;
  chatState.messages.push({role:'user', content:input});
  $('chatInput').value = '';
  renderChat();
  $('chatResult').textContent = 'Generating…';
  try {
    const data = await api('/api/chat',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({experiment, messages:chatState.messages, max_new_tokens:160, temperature:0.8})
    });
    const answer = (data.text || '').trim();
    chatState.messages.push({role:'assistant', content:answer || '(The model returned an empty response.)'});
    renderChat();
    $('chatResult').textContent = `Checkpoint epoch: ${data.epoch ?? 'latest'}`;
  } catch(e) {
    chatState.messages.pop();
    renderChat();
    $('chatResult').textContent = e.message;
  }
}

async function runAI(){
  const prompt = $('aiPrompt').value.trim();
  if(!prompt) return alert('Enter a prompt first.');
  const payload = {
    provider: $('aiProvider').value,
    model: $('aiModel').value.trim() || null,
    api_key: $('aiKey').value.trim() || null,
    prompt
  };
  try {
    const data = await api('/api/ai/generate', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(payload)
    });
    $('aiResult').textContent = `[${data.provider} / ${data.model}]\n\n${data.text}`;
  } catch(e) { $('aiResult').textContent = e.message; }
}

async function downloadModel(){
  const experiment = $('downloadExperiment').value.trim();
  if(!experiment) return alert('Enter the experiment name.');
  try {
    const key = $('apiKey').value.trim();
    const headers = key ? {'X-API-Key': key} : {};
    const response = await fetch('/api/experiments/'+encodeURIComponent(experiment)+'/download', {headers});
    if(!response.ok){
      const text = await response.text();
      let data; try { data = JSON.parse(text); } catch { data = {detail:text}; }
      throw new Error(data.detail || `HTTP ${response.status}`);
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${experiment}-trained-model.zip`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    $('downloadResult').textContent = 'Download started successfully.';
  } catch(e) { $('downloadResult').textContent = e.message; }
}
