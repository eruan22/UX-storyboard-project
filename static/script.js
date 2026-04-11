// load example button
document.getElementById("example-btn").addEventListener("click", () => {
    document.getElementById("persona").value = "Working professional";
    document.getElementById("goal").value = "Book a last minute flight for a business trip";
    document.getElementById("product").value = "Travel booking website";
    document.getElementById("scenario").value = "2 hours before departure";
});

document.getElementById("generate-btn").addEventListener("click", async () => {
    const persona  = document.getElementById("persona").value.trim();
    const goal     = document.getElementById("goal").value.trim();
    const product  = document.getElementById("product").value.trim();
    const scenario = document.getElementById("scenario").value.trim();

    if (!persona || !goal || !product || !scenario) {
        alert("Please fill in all fields.");
        return;
    }

    // show loading
    document.getElementById("loading").classList.remove("hidden");
    document.getElementById("storyboard-section").classList.add("hidden");
    document.getElementById("critique-section").classList.add("hidden");
    document.getElementById("recommendations-section").classList.add("hidden");

    const response = await fetch("/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ persona, goal, product, scenario })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value);
        const lines = text.split("\n").filter(l => l.startsWith("data: "));

        for (const line of lines) {
            const json_data = JSON.parse(line.replace("data: ", ""));

            if (json_data.type === "panels") {
                renderPanels(json_data.data);
                document.getElementById("storyboard-section").classList.remove("hidden");
                document.getElementById("loading").innerHTML = `
                    <div class="spinner"></div>
                    <p>Running UX Critic...</p>
                `;
            }

            if (json_data.type === "critiques") {
                renderCritiques(json_data.data);
                document.getElementById("critique-section").classList.remove("hidden");
                document.getElementById("loading").innerHTML = `
                    <div class="spinner"></div>
                    <p>Generating recommendations...</p>
                `;
            }

            if (json_data.type === "recommendations") {
                renderRecommendations(json_data.data);
                document.getElementById("recommendations-section").classList.remove("hidden");
            }

            if (json_data.type === "done") {
                document.getElementById("loading").classList.add("hidden");
            }
        }
    }
});

function renderPanels(panels) {
    document.getElementById("panels-container").innerHTML = panels.map(p => `
        <div class="panel-card">
            <h4>Panel ${p.panel_number}</h4>
            <p><strong>Action:</strong> ${p.action}</p>
            <p><strong>Context:</strong> ${p.context}</p>
            <p><strong>Emotion:</strong> ${p.emotion}</p>
        </div>
    `).join("");
}

function renderCritiques(critiques) {
    document.getElementById("critiques-container").innerHTML = critiques.map(c => `
        <div class="critique-card ${c.severity}">
            <span class="severity-badge">${c.severity}</span>
            <p><strong>Panel ${c.panel}:</strong> ${c.pain_point}</p>
            <p>${c.reason}</p>
        </div>
    `).join("");
}

function renderRecommendations(recs) {
    document.getElementById("recommendations-container").innerHTML = recs.map(r => `
        <div class="rec-card">
            <p><strong>Panel ${r.panel}:</strong> ${r.pain_point}</p>
            <p>${r.recommendation}</p>
        </div>
    `).join("");
}