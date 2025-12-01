document.getElementById("tag-btn").addEventListener("click", async () => {
    const input = document.getElementById("dogri-input").value.trim();
    const loading = document.getElementById("loading");
    const tableContainer = document.getElementById("results-table-container");
    const resultsBody = document.getElementById("results-body");

    if (!input) {
        alert("Please enter a Dogri sentence!");
        return;
    }

    loading.style.display = "block";
    tableContainer.classList.add("hidden");
    resultsBody.innerHTML = "";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ sentence: input })
        });

        const data = await response.json();
        loading.style.display = "none";

        if (data.result) {
            data.result.forEach(row => {
                const tr = document.createElement("tr");
                tr.innerHTML = `<td>${row.word}</td><td>${row.tag}</td>`;
                resultsBody.appendChild(tr);
            });
            tableContainer.classList.remove("hidden");
        } else {
            alert("Error: " + data.error);
        }
    } catch (err) {
        loading.style.display = "none";
        alert("Server error: " + err);
    }
});

document.getElementById("sample-btn").addEventListener("click", () => {
    document.getElementById("dogri-input").value = "अंश ते समर्थ इक परियोजना पर कम्म करा दे न ।";
});

document.getElementById("clear-btn").addEventListener("click", () => {
    document.getElementById("dogri-input").value = "";
    document.getElementById("results-body").innerHTML = "";
    document.getElementById("results-table-container").classList.add("hidden");
});
