const apiUrl = 'http://127.0.0.1:8000/predict';

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
});

document.getElementById('predict-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  const data = Object.fromEntries(new FormData(form).entries());
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
    showOutput(`<strong>Riesgo:</strong> ${pct}%<br><strong>Acción:</strong> ${action}`, pct > 50 ? 'high' : 'low');
  } catch (err) {
    showOutput('Error de red: ' + err, 'error');
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = originalText;
  }
});

function showOutput(html, level) {
  const out = document.getElementById('output');
  const div = document.createElement('div');
  div.className = 'result ' + level;
  div.innerHTML = html;
  out.innerHTML = '';
  out.appendChild(div);
}
