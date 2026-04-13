// load example button
document.getElementById("example-btn").addEventListener("click", () => {
    document.getElementById("persona").value = "Working professional";
    document.getElementById("goal").value = "Book a last minute flight for a business trip";
    document.getElementById("product").value = "Travel booking website";
    document.getElementById("scenario").value = "2 hours before departure";
});

// generate button
document.getElementById("generate-btn").addEventListener("click", async () => {
    const persona  = document.getElementById("persona").value.trim();
    const goal     = document.getElementById("goal").value.trim();
    const product  = document.getElementById("product").value.trim();
    const scenario = document.getElementById("scenario").value.trim();

    if (!persona || !goal || !product || !scenario) {
        alert("Please fill in all fields.");
        return;
    }

    // show loading, hide results
    document.getElementById("loading").classList.remove("hidden");
    document.getElementById("storyboard-section").classList.add("hidden");
    document.getElementById("critique-section").classList.add("hidden");
    document.getElementById("recommendations-section").classList.add("hidden");

    try {
        const response = await fetch("/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ persona, goal, product, scenario })
        });

        const data = await response.json();

        // render panels
        const panelsContainer = document.getElementById("panels-container");
        panelsContainer.innerHTML = data.panels.map(p => `
            <div class="panel-card">
                <h4>Panel ${p.panel_number}</h4>
                <p><strong>Action:</strong> ${p.action}</p>
                <p><strong>Context:</strong> ${p.context}</p>
                <p><strong>Emotion:</strong> ${p.emotion}</p>
            </div>
        `).join("");

        // render critiques
        const critiquesContainer = document.getElementById("critiques-container");
        critiquesContainer.innerHTML = data.critiques.map(c => `
            <div class="critique-card ${c.severity}">
                <span class="severity-badge">${c.severity}</span>
                <p><strong>Panel ${c.panel}:</strong> ${c.pain_point}</p>
                <p>${c.reason}</p>
            </div>
        `).join("");

        // render recommendations
        const recsContainer = document.getElementById("recommendations-container");
        recsContainer.innerHTML = data.recommendations.map(r => `
            <div class="rec-card">
                <p><strong>Panel ${r.panel}:</strong> ${r.pain_point}</p>
                <p>${r.recommendation}</p>
            </div>
        `).join("");

        // show results
        document.getElementById("storyboard-section").classList.remove("hidden");
        document.getElementById("critique-section").classList.remove("hidden");
        document.getElementById("recommendations-section").classList.remove("hidden");

    } catch (err) {
        alert("Something went wrong. Check the terminal for errors.");
        console.error(err);
    } finally {
        document.getElementById("loading").classList.add("hidden");
    }
});