// Progress polling for the submission page, plus the CSRF header for fetch calls.
(function () {
  var box = document.getElementById("progress");
  if (!box) return;
  var id = box.dataset.id, bar = document.getElementById("bar"), stage = document.getElementById("stage"), pct = document.getElementById("pct");
  function tick() {
    fetch("/api/submissions/" + encodeURIComponent(id), { credentials: "same-origin" })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        bar.style.width = d.progress + "%";
        pct.textContent = Math.round(d.progress) + "%";
        stage.textContent = d.stage;
        if (d.status === "done" || d.status === "failed") { window.location.reload(); return; }
        setTimeout(tick, 2000);
      })
      .catch(function () { setTimeout(tick, 5000); });
  }
  tick();
})();
