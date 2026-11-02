const $=id=>document.getElementById(id);
const pct=v=>(Number(v)*100).toFixed(2)+"%";
async function loadStats(){
 try{
  const r=await fetch("/api/stats");if(!r.ok)throw Error("Unable to load analytics");const d=await r.json();
  $("rows").textContent=Number(d.dataset_rows).toLocaleString();$("classes").textContent=d.class_count;
  $("accuracy").textContent=pct(d.best_metrics.accuracy);$("bestModel").textContent=d.best_model;
  $("selectedLabel").textContent=d.best_model;$("total").textContent=Number(d.dataset_rows).toLocaleString();
  const models=[...d.models].sort((a,b)=>b.accuracy-a.accuracy);
  $("bars").innerHTML=models.map(m=>`<div class="bar-row ${m.name===d.best_model?"best":""}"><span class="bar-name">${m.name}</span><div class="bar-track"><div class="bar-fill" style="width:${m.accuracy*100}%"></div></div><span class="bar-value">${pct(m.accuracy)}</span></div>`).join("");
  const metrics=[["Accuracy",d.best_metrics.accuracy],["Macro precision",d.best_metrics.precision_macro],["Macro recall",d.best_metrics.recall_macro],["Macro F1-score",d.best_metrics.f1_macro]];
  $("scores").innerHTML=metrics.map(([n,v])=>`<div class="score"><span>${n}</span><div class="score-track"><i style="width:${v*100}%"></i></div><strong>${pct(v)}</strong></div>`).join("");
  $("chips").innerHTML=Object.entries(d.class_counts).map(([n,c])=>`<span class="chip">${n}<b>${c}</b></span>`).join("");
//   drawMatrix(d.class_names,d.confusion_matrix);
 }catch(e){console.error(e);$("bestModel").textContent="Unavailable";$("bars").textContent="Could not load dashboard data. Is the Flask server running?"}
}
function drawMatrix(labels,data){
 const host=$("matrix"),n=labels.length;host.style.gridTemplateColumns=`56px repeat(${n},22px)`;
 let out='<div class="cell label"></div>'+labels.map(x=>`<div class="cell label" title="${x}">${x.slice(0,3)}</div>`).join("");
 for(let i=0;i<n;i++){out+=`<div class="cell label" title="${labels[i]}">${labels[i].slice(0,8)}</div>`;
  const max=Math.max(1,...data[i]);for(let j=0;j<n;j++){let v=data[i][j],diag=i===j;
   let bg=diag?`rgba(65,126,79,${.25+.75*v/max})`:v?"#f5e8e5":"#f3f6f3";
   out+=`<div class="cell" style="background:${bg};color:${diag?"#fff":"#9b766e"}" title="Actual ${labels[i]} / predicted ${labels[j]}: ${v}">${v||""}</div>`;
  }}
 host.innerHTML=out;
}
$("form").addEventListener("submit",async e=>{
 e.preventDefault();$("error").classList.remove("show");
 const form=e.currentTarget;if(!form.reportValidity())return;
 const data=Object.fromEntries(new FormData(form).entries());
 for(const k in data){if(!Number.isFinite(Number(data[k]))){$("error").textContent=`${k} must be a valid number.`;$("error").classList.add("show");return}data[k]=Number(data[k])}
 const btn=$("submit");btn.disabled=true;btn.textContent="Analysing measurements…";
 try{
  const r=await fetch("/api/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)});
  const d=await r.json();if(!r.ok)throw Error(d.error||"Prediction failed");
  $("empty").classList.add("hidden");$("result").classList.remove("hidden");
  $("crop").textContent=d.crop[0].toUpperCase()+d.crop.slice(1);$("used").textContent=d.model;
  const top=d.top_predictions||[],p=top.length?top[0].probability:null;
  $("prob").textContent=p===null?"Not available":pct(p);$("probbar").style.width=p===null?"0%":`${p*100}%`;
  $("alternatives").innerHTML=top.slice(1).map(x=>`<div class="alternative"><span>${x.crop}</span><b>${pct(x.probability)}</b></div>`).join("");
  $("result").scrollIntoView({behavior:"smooth",block:"nearest"});
 }catch(err){$("error").textContent=err.message;$("error").classList.add("show")}
 finally{btn.disabled=false;btn.innerHTML="Generate recommendation <b>↗</b>"}
});
$("form").addEventListener("reset",()=>{$("error").classList.remove("show")});
// $("today").textContent=new Intl.DateTimeFormat(undefined,{day:"2-digit",month:"short",year:"numeric"}).format(new Date());
loadStats();
