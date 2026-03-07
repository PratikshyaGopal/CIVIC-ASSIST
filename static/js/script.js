// ==================== MODAL FUNCTIONALITY ====================

/**
 * Open the update modal and populate it with complaint data
 * @param {number} complaintId - The ID of the complaint to update
 * @param {string} currentStatus - Current status of the complaint
 * @param {string} currentDepartment - Current department assigned
 * @param {string} currentRemarks - Current remarks
 */
function openModal(complaintId, currentStatus, currentDepartment, currentRemarks) {
    const modal = document.getElementById('updateModal');
    const form = document.getElementById('updateForm');

    // Set form action URL
    form.action = `/admin/update_complaint/${complaintId}`;

    // Populate form fields with current values
    document.getElementById('status').value = currentStatus;
    document.getElementById('department').value = currentDepartment;
    document.getElementById('remarks').value = currentRemarks;

    // Display modal
    modal.style.display = 'block';
}

/**
 * Close the update modal
 */
function closeModal() {
    const modal = document.getElementById('updateModal');
    modal.style.display = 'none';
}

// Close modal when clicking outside of it
window.onclick = function (event) {
    const modal = document.getElementById('updateModal');
    if (event.target == modal) {
        modal.style.display = 'none';
    }
}

// Module-level reference so we can update the map on repeated clicks
let _complaintMap = null;
let _complaintMarker = null;

/**
 * Get user's current location using browser's Geolocation API
 * and render it on an embedded Leaflet map
 */
function getLocation() {
    const locationStatus = document.getElementById('location-status');
    const locationDisplay = document.getElementById('location-display');
    const locationIcon = document.getElementById('location-icon');

    if (!locationStatus || !locationDisplay || !locationIcon) {
        console.error('Geolocation UI elements not found');
        return;
    }

    // Check if geolocation is supported
    if (!navigator.geolocation) {
        locationStatus.innerHTML = '<span style="color: red;">❌ Geolocation not supported</span>';
        return;
    }

    // Show loading state
    locationStatus.innerHTML = '<span style="color: #f59e0b;">⏳ Detecting location...</span>';
    locationIcon.textContent = 'hourglass_empty';

    // Get current position
    navigator.geolocation.getCurrentPosition(
        // Success callback
        function (position) {
            const latitude = position.coords.latitude;
            const longitude = position.coords.longitude;

            // Store coordinates in hidden inputs
            var latInput = document.getElementById('latitude');
            var lonInput = document.getElementById('longitude');
            if (latInput) latInput.value = latitude;
            if (lonInput) lonInput.value = longitude;

            // Update status icon
            locationStatus.innerHTML = '<span style="color: #10b981;">✅ Location detected</span>';
            locationIcon.textContent = 'check_circle';
            locationIcon.style.color = '#10b981';
            locationDisplay.innerHTML =
                'Lat: ' + latitude.toFixed(6) + ', Lon: ' + longitude.toFixed(6) +
                '<br><a href="https://www.google.com/maps?q=' + latitude + ',' + longitude + '" target="_blank" style="color: #137fec;">📍 View on Google Maps</a>';

            // ── Show the Leaflet map ──────────────────────────────────────
            const placeholder = document.getElementById('map-placeholder');
            const mapDiv      = document.getElementById('complaint-map');

            if (mapDiv) {
                // Hide placeholder, show map
                if (placeholder) placeholder.style.display = 'none';
                mapDiv.style.display = 'block';

                if (!_complaintMap) {
                    // First call: initialize the map
                    _complaintMap = L.map('complaint-map', { zoomControl: true, scrollWheelZoom: false });
                    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                        attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
                        maxZoom: 19
                    }).addTo(_complaintMap);
                }

                // Set view and add/move marker
                _complaintMap.setView([latitude, longitude], 15);
                if (_complaintMarker) {
                    _complaintMarker.setLatLng([latitude, longitude]);
                } else {
                    _complaintMarker = L.marker([latitude, longitude])
                        .addTo(_complaintMap)
                        .bindPopup('<b>Your complaint location</b><br>Lat: ' + latitude.toFixed(5) + '<br>Lon: ' + longitude.toFixed(5))
                        .openPopup();
                }

                // Leaflet needs a size refresh when revealed from display:none
                setTimeout(function() { _complaintMap.invalidateSize(); }, 100);

                // Optional: reverse geocode with Nominatim (free, no API key)
                fetch('https://nominatim.openstreetmap.org/reverse?format=json&lat=' + latitude + '&lon=' + longitude)
                    .then(function(r) { return r.json(); })
                    .then(function(data) {
                        if (data && data.display_name) {
                            const addr = data.display_name;
                            if (_complaintMarker) {
                                _complaintMarker.setPopupContent('<b>Your complaint location</b><br><small>' + addr + '</small>');
                                _complaintMarker.openPopup();
                            }
                            locationDisplay.innerHTML =
                                '📍 <strong>' + addr.split(',').slice(0,3).join(', ') + '</strong>' +
                                '<br><small>Lat: ' + latitude.toFixed(6) + ', Lon: ' + longitude.toFixed(6) + '</small>' +
                                '<br><a href="https://www.google.com/maps?q=' + latitude + ',' + longitude + '" target="_blank" style="color: #137fec;">View on Google Maps</a>';
                        }
                    })
                    .catch(function() { /* silently ignore reverse geocode failures */ });
            }
            // ─────────────────────────────────────────────────────────────
        },
        // Error callback
        function (error) {
            let errorMessage = '';
            switch (error.code) {
                case error.PERMISSION_DENIED:
                    errorMessage = '❌ Location permission denied';
                    break;
                case error.POSITION_UNAVAILABLE:
                    errorMessage = '❌ Location unavailable';
                    break;
                case error.TIMEOUT:
                    errorMessage = '❌ Location request timeout';
                    break;
                default:
                    errorMessage = '❌ Location error';
            }
            locationStatus.innerHTML = '<span style="color: red;">' + errorMessage + '</span>';
            locationIcon.textContent = 'location_on';
        },
        // Options
        {
            enableHighAccuracy: true,
            timeout: 10000,
            maximumAge: 0
        }
    );
}


// ==================== IMAGE PREVIEW FUNCTIONALITY ====================

/**
 * Preview selected image before upload
 * @param {Event} event - File input change event
 */
function previewImage(event) {
    const file = event.target.files[0];
    const preview = document.getElementById('image-preview');

    // Clear previous preview
    preview.innerHTML = '';

    if (file) {
        // Validate file size (5MB max)
        if (file.size > 5 * 1024 * 1024) {
            preview.innerHTML = '<span class="location-error">❌ File size must be less than 5MB</span>';
            event.target.value = '';
            return;
        }

        // Validate file type
        const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif'];
        if (!validTypes.includes(file.type)) {
            preview.innerHTML = '<span class="location-error">❌ Invalid file type. Use PNG, JPG, JPEG, or GIF</span>';
            event.target.value = '';
            return;
        }

        // Create preview
        const reader = new FileReader();
        reader.onload = function (e) {
            preview.innerHTML = `
                <div style="margin-top: 1rem;">
                    <strong>Image Preview:</strong><br>
                    <img src="${e.target.result}" alt="Preview" style="max-width: 300px; max-height: 300px; margin-top: 0.5rem; border-radius: 5px; border: 2px solid var(--border-color);">
                    <br>
                    <small style="color: var(--gray);">${file.name} (${(file.size / 1024).toFixed(2)} KB)</small>
                </div>
            `;
        };
        reader.readAsDataURL(file);
    }
}

// ==================== FORM VALIDATION ====================

/**
 * Add basic client-side validation to forms
 */
document.addEventListener('DOMContentLoaded', function () {
    // Auto-hide alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            alert.style.opacity = '0';
            alert.style.transition = 'opacity 0.5s';
            setTimeout(() => alert.remove(), 500);
        }, 5000);
    });

    // Add smooth scrolling for anchor links (only for anchors with actual section targets)
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const href = this.getAttribute('href');
            if (href && href.length > 1) {  // Only for #section-id, not bare "#"
                e.preventDefault();
                const target = document.querySelector(href);
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth'
                    });
                }
            }
        });
    });
});

// ==================== UTILITY FUNCTIONS ====================

/**
 * Confirm before logout
 */
function confirmLogout() {
    return confirm('Are you sure you want to logout?');
}