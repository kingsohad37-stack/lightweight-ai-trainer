/* TrainItLocal automatic two-step workflow. */
(() => {
  const $ = id => document.getElementById(id);
  if (!$('autoPrompt') || !$('autoBuild')) return;
  const api = async (path, options={}) => {
    const key = $('apiKey')?.value.trim();
    const headers = {'Content-Type':'application/json', ...(options.headers||{})}; if(key) headers['X-API-Key']=key;
    const r=await fetch(path,{...options,headers}); const text=await r.text(); let data; try{data=text?JSON.parse(text):{}}catch{data={detail:text}};
    if(!r.ok) throw new Error(data.detail||data.error||`Request failed (${r.status})`); return data;
  };
  const status=s=>$('autoStatus').textContent=s, result=s=>$('autoResult').textContent=s;
  $('autoBuild').addEventListener('click',async()=>{
    const description=$('autoPrompt').value.trim(); if(!description)return status('Describe the AI you want first.');
    $('autoBuild').disabled=true; $('autoDownload').hidden=true;
    try{
      status('Understanding what you want…'); const planned=await api('/api/auto/plan',{method:'POST',body:JSON.stringify({description})}); result(JSON.stringify(planned.plan,null,2));
      status('Generating training data and building your AI…'); const job=await api('/api/auto/build',{method:'POST',body:JSON.stringify({description,plan:planned.plan})});
      while(true){
        const s=await api('/api/auto/status/'+encodeURIComponent(job.job_id)); status(s.message||s.status||'Working…');
        if(s.status==='completed'){
          const d=s.result||{}; result(JSON.stringify(d,null,2)); $('autoDownload').hidden=false;
          $('autoDownload').onclick=async()=>{try{status('Preparing your download…'); const key=$('apiKey')?.value.trim(); const h=key?{'X-API-Key':key}:{}; const r=await fetch(d.download_url,{headers:h}); if(!r.ok)throw new Error((await r.text())||`HTTP ${r.status}`); const b=await r.blob(),u=URL.createObjectURL(b),a=document.createElement('a'); a.href=u;a.download=`${d.experiment||'trainitlocal-ai'}-trained-model.zip`;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(u),60000);status('Your deployable AI is ready — download started.')}catch(e){status('Download failed');result(e.message)}};
          status('Your deployable AI is ready.'); break;
        }
        if(s.status==='failed')throw new Error(s.error||'Automatic build failed.'); await new Promise(r=>setTimeout(r,1500));
      }
    }catch(e){status('Build failed');result(e.message)}finally{$('autoBuild').disabled=false}
  });
})();
