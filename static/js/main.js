function animateCounter(id, end) {
    let elem = document.getElementById(id);
    let start = 0;
    let duration = 1000;
    let step = Math.ceil(end / (duration / 16));
    let interval = setInterval(() => {
        start += step;
        if (start >= end) {
            start = end;
            clearInterval(interval);
        }
        elem.textContent = start.toLocaleString();
    }, 16);
}
