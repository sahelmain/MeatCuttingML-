function [features] = process_radar_data(radar_data_path)
    % Process millimeter radar data and extract relevant features
    %
    % Args:
    %   radar_data_path: Path to the radar data file
    %
    % Returns:
    %   features: Extracted features from radar measurements

    % Load radar data
    radar_data = load(radar_data_path);

    % Signal processing
    [range_profile, doppler_profile] = process_radar_signal(radar_data);

    % Feature extraction
    features = struct();
    features.morphology = extract_morphological_features(range_profile);
    features.motion = extract_motion_features(doppler_profile);
    features.surface_metrics = calculate_surface_metrics(range_profile);
end

function [range_profile, doppler_profile] = process_radar_signal(radar_data)
    % Process raw radar signal

    % Range processing
    range_fft = fft(radar_data.samples);
    range_profile = abs(range_fft);

    % Doppler processing
    doppler_fft = fft(radar_data.samples, [], 2);
    doppler_profile = abs(doppler_fft);

    % Apply windowing and noise reduction
    range_profile = apply_window(range_profile);
    doppler_profile = apply_window(doppler_profile);
end

function windowed = apply_window(signal)
    % Apply Hamming window for sidelobe reduction
    window = hamming(length(signal));
    windowed = signal .* window;
end

function features = extract_morphological_features(range_profile)
    % Extract morphological features from range profile
    features = struct();

    % Calculate basic statistics
    features.mean_range = mean(range_profile);
    features.std_range = std(range_profile);

    % Find peaks for surface detection
    [peaks, locations] = findpeaks(range_profile, 'MinPeakHeight', mean(range_profile));
    features.num_surfaces = length(peaks);
    features.surface_distances = diff(locations);
end

function features = extract_motion_features(doppler_profile)
    % Extract motion-related features from Doppler profile
    features = struct();

    % Calculate velocity statistics
    velocities = calculate_velocities(doppler_profile);
    features.mean_velocity = mean(velocities);
    features.velocity_spread = std(velocities);
end

function features = calculate_surface_metrics(range_profile)
    % Calculate surface-related metrics
    features = struct();

    % Surface roughness estimation
    features.roughness = estimate_roughness(range_profile);

    % Surface area estimation
    features.surface_area = estimate_surface_area(range_profile);
end

function velocities = calculate_velocities(doppler_profile)
    % Convert Doppler shifts to velocities
    freq_resolution = 1; % Hz per bin
    wavelength = 0.00375; % For 77 GHz radar
    velocities = doppler_profile * (wavelength/2) * freq_resolution;
end

function roughness = estimate_roughness(range_profile)
    % Estimate surface roughness from range profile variations
    roughness = std(diff(range_profile));
end

function area = estimate_surface_area(range_profile)
    % Estimate surface area from range profile
    % This is a simplified estimation
    area = trapz(range_profile);
end
