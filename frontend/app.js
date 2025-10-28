// Señal de arranque para confirmar carga del JS
console.info('[frontend] app.js cargado');

// Permitir modo debug desde la URL del navegador: ?debug=1
const urlParams = new URLSearchParams(window.location.search);
const DEBUG = urlParams.get('debug') === '1';
const apiUrl = DEBUG
  ? 'http://127.0.0.1:8000/predict?debug=true'
  : 'http://127.0.0.1:8000/predict';

// Si la URL contiene ?blank=1, limpiar valores del formulario para presentarlo vacío
window.addEventListener('DOMContentLoaded', () => {
  try {
    const params = new URLSearchParams(window.location.search);
    if (params.get('blank') === '1') {
      const namesToClear = ['age','height_cm','weight_kg','liver_function_score','alpha_fetoprotein_level'];
      namesToClear.forEach(n => {
        const el = document.querySelector(`[name="${n}"]`);
        if (el) el.value = '';
      });
      // Dejar selects en su primera opción (generalmente 'No' / 'Masculino')
      const selects = ['hepatitis_b','hepatitis_c','cirrhosis_history','family_history_cancer','diabetes','gender','alcohol_consumption','smoking_status','physical_activity_level'];
      selects.forEach(n => {
        const s = document.querySelector(`[name="${n}"]`);
        if (s) s.selectedIndex = 0;
      });
    }
  } catch (e) {
    // no crítico, seguir sin bloquear
    console.warn('Error al procesar parámetro blank:', e);
  }
  console.debug('[frontend] DOMContentLoaded disparado');
});

// Asegurar el binding del submit después de que el DOM esté listo
document.addEventListener('DOMContentLoaded', () => {
  const formEl = document.getElementById('predict-form');
  if (!formEl) {
    console.error('[frontend] No se encontró el formulario con id="predict-form"');
    return;
  }
  console.debug('[frontend] Handler de submit conectado');

  formEl.addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  const data = Object.fromEntries(new FormData(form).entries());
  // Guardamos copia del payload que enviaremos al backend (útil para debug y logging)
  let debugPayload = null;
  // Numeric conversions and sensible defaults for optional lab fields
  data.age = Number(data.age);
  data.height_cm = Number(data.height_cm);
  data.weight_kg = Number(data.weight_kg);
  data.liver_function_score = data.liver_function_score === undefined || data.liver_function_score === '' ? 0.0 : Number(data.liver_function_score);
  data.alpha_fetoprotein_level = data.alpha_fetoprotein_level === undefined || data.alpha_fetoprotein_level === '' ? 0.0 : Number(data.alpha_fetoprotein_level);
  data.hepatitis_b = Number(data.hepatitis_b);
  data.hepatitis_c = Number(data.hepatitis_c);
  data.cirrhosis_history = Number(data.cirrhosis_history);
  data.family_history_cancer = Number(data.family_history_cancer);
  data.diabetes = Number(data.diabetes);

  // Capturamos exactamente lo que enviaremos en el body (siempre)
  debugPayload = JSON.parse(JSON.stringify(data));
  // Log SIEMPRE del payload que se enviará al endpoint
  console.log('Payload enviado (formulario):', debugPayload);

  const submitBtn = form.querySelector('button[type="submit"]');
  submitBtn.disabled = true;
  const originalText = submitBtn.textContent;
  submitBtn.textContent = 'Calculando...';
  try {
    const res = await fetch(apiUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });

    if (!res.ok) {
      const text = await res.text();
      showOutput('Error: ' + text, 'error');
      return;
    }

    const json = await res.json();
    const pct = json.risk_pct;
    const action = json.action;
    let extra = '';
    if (DEBUG) {
      // Payload enviado
      if (debugPayload) {
        const prettySent = JSON.stringify(debugPayload, null, 2);
        extra += `<div><em>Payload enviado (formulario):</em></div><pre class=\"debug-block\">${prettySent}</pre>`;
      }
      // Entrada normalizada por el backend (antes del preprocesamiento/modelo)
      if (json.normalized_input) {
        const prettyNorm = JSON.stringify(json.normalized_input, null, 2);
        extra += `<div><em>Entrada normalizada (backend):</em></div><pre class=\"debug-block\">${prettyNorm}</pre>`;
      }
      // También mostramos toda la respuesta para depuración
      console.log('Respuesta /predict:', json);
    }
    showOutput(`
      <strong>Riesgo:</strong> ${pct}%<br>
      <strong>Acción:</strong> ${action}
      ${extra ? '<br><strong>Entrada normalizada:</strong><br>' + extra : ''}
    `, pct > 50 ? 'high' : 'low');
  } catch (err) {
    showOutput('Error de red: ' + err, 'error');
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = originalText;
  }
  });
});

function showOutput(html, level) {
  const out = document.getElementById('output');
  const div = document.createElement('div');
  div.className = 'result ' + level;
  div.innerHTML = html;
  out.innerHTML = '';
  out.appendChild(div);
}
