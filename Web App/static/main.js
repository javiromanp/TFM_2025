ruta_json = '/static/data/homomex_frasestrain.json'
// ruta_json = '/static/data/homomex_frasesprueba.json'

let categories = {};
let categoriesLoaded = false;

function transformarEtiqueta(clave) {
    const map = {
        npDirectoColectivo: "Bienvenida directa a la comunidad LGBTQ+",
        npIndirectoColectivo: "Bienvenida indirecta a la comunidad LGBTQ+",
        npDirectoSubgrupos: "Apoyo directo a personas gays, lesbianas, trans…",
        npIndirectoSubgrupos: "Apoyo indirecto a personas gays, lesbianas, trans…",
        pDirectoColectivo: "Rechazo directo a la comunidad LGBTQ+",
        pIndirectoColectivo: "Rechazo indirecto a la comunidad LGBTQ+",
        pDirectoSubgrupos: "Rechazo directo a personas gays, lesbianas, trans…",
        pIndirectoSubgrupos: "Rechazo indirecto a personas gays, lesbianas, trans…",
        nrEjemplos: "Frases sin relación con temas LGBTQ+"
    };
    return map[clave] || clave;
}

function conectarBotonesCategorias() {
    const botones = document.querySelectorAll('[data-category]');
    botones.forEach(btn => {
        btn.addEventListener('click', () => {
            const clave = btn.dataset.category;
            if (!categoriesLoaded) {
                alert("Frases aún no cargadas.");
                return;
            }
            const textarea = document.getElementById('text-input');
            textarea.value = categories[clave]?.join('\n') || 'Categoría no encontrada.';
        });
    });
}

window.addEventListener('load', () => {
    const buttons = document.querySelectorAll('.example-buttons button');
    buttons.forEach(btn => btn.disabled = true);

    fetch(ruta_json)
        .then(response => response.json())
        .then(data => {
            categories = data;
            categoriesLoaded = true;

            const buttons = document.querySelectorAll('.example-buttons button');
            buttons.forEach(btn => btn.disabled = false);

            const container = document.getElementById('dynamic-buttons');
            container.innerHTML = '';

            const grupos = {
                np: { label: "Clase NP — No fóbico", color: "green" },
                p:  { label: "Clase P — Fóbico", color: "red" },
                nr: { label: "Clase NR — No relacionado", color: "yellow" }
            };

            for (const key of Object.keys(categories)) {
                const prefix = key.toLowerCase().startsWith("np") ? "np"
                    : key.toLowerCase().startsWith("p")  ? "p"
                    : key.toLowerCase().startsWith("nr") ? "nr"
                    : null;
                const grupo = grupos[prefix];
                if (!grupo) continue;

                if (!container.querySelector(`.group-${prefix}`)) {
                    const header = document.createElement('h3');
                    header.textContent = grupo.label;
                    header.classList.add(grupo.color, `group-${prefix}`);
                    container.appendChild(header);

                    const wrapper = document.createElement('div');
                    wrapper.classList.add('example-buttons', grupo.color, `buttons-${prefix}`);
                    container.appendChild(wrapper);
                }

                const button = document.createElement('button');
                button.dataset.category = key;
                button.textContent = transformarEtiqueta(key);
                container.querySelector(`.buttons-${prefix}`).appendChild(button);
            }

            conectarBotonesCategorias();
        })
        .catch(err => {
            console.error("No se pudieron cargar las frases de ejemplo:", err);
        });


    fetch('/device')
        .then(res => res.json())
        .then(data => {
            const indicator = document.getElementById('device-indicator');
            if (data.device === "cuda") {
                indicator.textContent = "Dispositivo: GPU";
                indicator.classList.add("gpu");
            } else {
                indicator.textContent = "Dispositivo: CPU";
                indicator.classList.add("cpu");
            }
        })
        .catch(() => {
            const indicator = document.getElementById('device-indicator');
            indicator.textContent = "Desconocido";
        });
});

function setExampleText(category) {
    const textarea = document.getElementById('text-input');
    if (!categoriesLoaded) {
        alert("Frases de ejemplo aún no cargadas. Intenta de nuevo en un momento.");
        return;
    }
    textarea.value = categories[category]?.join('\n') || 'Categoría no encontrada.';
}

document.getElementById('text-form').addEventListener('submit', async function (e) {
    e.preventDefault();

    const textInput = document.getElementById('text-input').value;
    let phrases = textInput.split('\n').filter(phrase => phrase.trim() !== '');

    if (phrases.length > 10) {
        alert('Por favor, introduce hasta 10 frases.');
        return;
    }

    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: phrases.join('\n') })
    });

    const data = await response.json();

    if (response.ok) {
        const resultsTable = document.getElementById('results-table').getElementsByTagName('tbody')[0];
        resultsTable.innerHTML = '';

        phrases.forEach((phrase, index) => {
            const row = resultsTable.insertRow();
            const cellIndex = row.insertCell(0);
            const cellText = row.insertCell(1);
            const cellClass = row.insertCell(2);
            const cellAction = row.insertCell(3);

            cellIndex.textContent = index + 1;
            cellText.textContent = phrase;

            const classValue = data.classes[index];
            const labelMap = {
                0: 'No fóbico',
                1: 'Fóbico',
                2: 'No relacionado'
            };
            cellClass.textContent = labelMap[classValue] || 'Desconocido';
            cellClass.className = `class-${classValue}`;

            const button = document.createElement('button');
            button.textContent = 'Ver resultado';
            button.onclick = function () {
                updateChart(data.scores[index], index + 1);
            };
            cellAction.appendChild(button);
        });
    } else {
        alert(data.error || 'Error al predecir las clases.');
    }
});

function updateChart(scores, tweetIndex) {
    const ctx = document.getElementById('results-chart').getContext('2d');
    if (window.resultsChart) {
        window.resultsChart.destroy();
    }
    window.resultsChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['No fóbico', 'Fóbico', 'No relacionado'],
            datasets: [{
                data: scores,
                backgroundColor: ['#28a745', '#dc3545', '#a18c00']
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: `Resultados para el tweet #${tweetIndex}`
                }
            }
        }
    });
}
