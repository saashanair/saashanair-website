document.addEventListener('DOMContentLoaded', () => {
  const el = document.getElementById('travel-map');
  if (!el) return;

  const places = JSON.parse(el.dataset.travels);
  const map = L.map(el).setView([20, 0], 2);

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxZoom: 18
  }).addTo(map);

  const markers = places.map((place, index) => {
    const hasNote = Boolean(place.url);
    const marker = L.marker([place.lat, place.lng], { icon: pinIcon(hasNote, index) });

    marker.bindPopup(popupHTML(place, hasNote), { maxWidth: 280 });
    marker.bindTooltip(tooltipHTML(place, hasNote), { direction: 'top', className: 'travel-tooltip' });
    marker.on('click', () => {
      const targetZoom = Math.max(map.getZoom(), 6);
      // Center on a point above the pin (in pixel space) so the pin lands in
      // the lower half of the map, leaving room for the popup above it.
      const targetPoint = map.project(marker.getLatLng(), targetZoom).subtract([0, 150]);
      const targetLatLng = map.unproject(targetPoint, targetZoom);

      map.closePopup();
      map.once('moveend', () => marker.openPopup());
      map.flyTo(targetLatLng, targetZoom, { duration: 0.8 });
    });

    return marker.addTo(map);
  });

  map.on('popupopen', (e) => initCarousel(e.popup.getElement()));

  if (markers.length) {
    map.fitBounds(L.featureGroup(markers).getBounds().pad(0.2));
  }
});

function pinIcon(hasNote, index) {
  const color = hasNote ? '#3b82f6' : '#94a3b8';
  const delay = Math.min(index * 60, 600);

  return L.divIcon({
    className: 'travel-pin',
    html: `<svg width="22" height="30" viewBox="0 0 30 40" style="animation-delay:${delay}ms">
      <path d="M15 0C6.7 0 0 6.7 0 15c0 11.3 15 25 15 25s15-13.7 15-25C30 6.7 23.3 0 15 0z" fill="${color}" stroke="#fff" stroke-width="2"/>
      <circle cx="15" cy="15" r="6" fill="#fff"/>
    </svg>`,
    iconSize: [22, 30],
    iconAnchor: [11, 30],
    popupAnchor: [0, -26],
    tooltipAnchor: [0, -24]
  });
}

function tooltipHTML(place, hasNote) {
  const dot = hasNote ? '<span class="travel-tooltip__dot"></span>' : '';
  return `${dot}${escapeHTML(place.name)}`;
}

function popupHTML(place, hasNote) {
  const header = `
    <div class="travel-popup__header">
      <strong class="travel-popup__title">${escapeHTML(place.name)}</strong>
      <span class="travel-popup__country">${escapeHTML(place.country)}</span>
    </div>`;

  if (!hasNote) {
    return `<div class="travel-popup">${header}<p class="travel-popup__visited">Visited 🧳</p></div>`;
  }

  const images = place.images || [];
  const carousel = images.length ? carouselHTML(place.name, images) : '';

  return `
    <div class="travel-popup">
      ${header}
      ${carousel}
      <p class="travel-popup__summary">${escapeHTML(place.summary)}</p>
      <a class="travel-popup__link" href="${escapeHTML(place.url)}">Read full recommendations &rarr;</a>
    </div>`;
}

function carouselHTML(name, images) {
  const slides = images.map((src) => `<img src="${escapeHTML(src)}" alt="${escapeHTML(name)}">`).join('');

  const nav =
    images.length > 1
      ? `<button class="travel-popup__nav travel-popup__nav--prev" aria-label="Previous photo">&#8249;</button>
         <button class="travel-popup__nav travel-popup__nav--next" aria-label="Next photo">&#8250;</button>
         <div class="travel-popup__dots">${images
           .map((_, i) => `<span class="travel-popup__dot${i === 0 ? ' is-active' : ''}"></span>`)
           .join('')}</div>`
      : '';

  return `<div class="travel-popup__carousel"><div class="travel-popup__slides">${slides}</div>${nav}</div>`;
}

function initCarousel(popupEl) {
  const carousel = popupEl.querySelector('.travel-popup__carousel');
  if (!carousel) return;

  const slides = carousel.querySelector('.travel-popup__slides');
  const dots = [...carousel.querySelectorAll('.travel-popup__dot')];
  let index = 0;

  const show = (i) => {
    index = (i + dots.length) % dots.length;
    slides.style.transform = `translateX(-${index * 100}%)`;
    dots.forEach((dot, d) => dot.classList.toggle('is-active', d === index));
  };

  carousel.querySelector('.travel-popup__nav--prev')?.addEventListener('click', () => show(index - 1));
  carousel.querySelector('.travel-popup__nav--next')?.addEventListener('click', () => show(index + 1));
}

function escapeHTML(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}
