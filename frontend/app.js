async function loadData() {
  const res = await fetch("../output/regime.json");
  const data = await res.json();

  const regimeCard = document.getElementById("regime-card");

  // Set text
  regimeCard.innerText = data.regime.toUpperCase();

  // Color logic
  if (data.regime.includes("bull")) {
    regimeCard.className += " bg-green-600";
  } else if (data.regime.includes("bear")) {
    regimeCard.className += " bg-red-600";
  } else {
    regimeCard.className += " bg-yellow-600";
  }

  // Metrics
  document.getElementById("ret").innerText = data.ret_21d.toFixed(3);
  document.getElementById("vol").innerText = data.vol_ratio.toFixed(2);
  document.getElementById("price").innerText = data.close;
}

loadData();