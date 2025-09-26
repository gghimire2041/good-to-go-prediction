function q(id){return document.getElementById(id)}

async function callApi(endpoint, payload){
  const base = q('apiBase').value.trim();
  const url = `${base}${endpoint}`;
  const headers = {'Content-Type':'application/json'};
  const token = q('authToken').value.trim();
  if (token) {
    // If token already starts with 'Bearer ' leave it; otherwise prepend
    headers['Authorization'] = token.toLowerCase().startsWith('bearer ') ? token : `Bearer ${token}`;
  }
  const r = await fetch(url, {method:'POST', headers, body: JSON.stringify(payload)});
  if(!r.ok){
    const t = await r.text();
    throw new Error(`${r.status} ${t}`);
  }
  return r.json();
}

function buildPayload(){
  return {
    gid: q('gid').value || undefined,
    risk_category: q('risk_category').value,
    region: q('region').value,
    business_type: q('business_type').value,
    regulatory_status: q('regulatory_status').value,
    compliance_level: q('compliance_level').value,
    revenue: Number(q('revenue').value),
    employee_count: Number(q('employee_count').value),
    years_in_business: Number(q('years_in_business').value),
    credit_score: Number(q('credit_score').value),
    debt_to_equity_ratio: Number(q('debt_to_equity_ratio').value),
    liquidity_ratio: Number(q('liquidity_ratio').value),
    market_share: Number(q('market_share').value),
    growth_rate: Number(q('growth_rate').value),
    profitability_margin: Number(q('profitability_margin').value),
    customer_satisfaction: Number(q('customer_satisfaction').value),
    operational_efficiency: Number(q('operational_efficiency').value),
    regulatory_violations: Number(q('regulatory_violations').value),
    audit_score: Number(q('audit_score').value),
    description: q('description').value
  };
}

async function onPredict(){
  q('result').textContent = 'Running...';
  try {
    const data = await callApi('/predict', buildPayload());
    q('result').textContent = JSON.stringify(data, null, 2);
  } catch(err){
    q('result').textContent = String(err);
  }
}

async function onExplain(){
  q('result').textContent = 'Running...';
  try {
    const data = await callApi('/explain', buildPayload());
    q('result').textContent = JSON.stringify(data, null, 2);
  } catch(err){
    q('result').textContent = String(err);
  }
}

async function onCheckHealth(){
  const base = q('apiBase').value.trim();
  try{
    const r = await fetch(`${base}/health`);
    const txt = await r.text();
    q('healthStatus').textContent = r.ok ? 'Healthy' : `Error: ${r.status}`;
  } catch(e){
    q('healthStatus').textContent = 'Unreachable';
  }
}

document.addEventListener('DOMContentLoaded', ()=>{
  q('btnPredict').addEventListener('click', onPredict);
  q('btnExplain').addEventListener('click', onExplain);
  q('checkHealth').addEventListener('click', onCheckHealth);
});
