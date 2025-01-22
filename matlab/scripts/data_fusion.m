function [fused_features] = data_fusion(ct_features, radar_features)
    % Fuse features from CT and radar data
    %
    % Args:
    %   ct_features: Features extracted from CT scans
    %   radar_features: Features extracted from radar measurements
    %
    % Returns:
    %   fused_features: Combined feature set

    % Initialize fused feature structure
    fused_features = struct();

    % Combine volume measurements
    fused_features.volume = combine_volume_measurements(...
        ct_features.volume, ...
        radar_features.surface_metrics.surface_area);

    % Combine tissue composition
    fused_features.composition = ct_features.tissue_ratios;

    % Combine surface metrics
    fused_features.surface = combine_surface_metrics(...
        ct_features.shape_metrics, ...
        radar_features.surface_metrics);

    % Add motion features from radar
    fused_features.motion = radar_features.motion;

    % Calculate confidence metrics
    fused_features.confidence = calculate_confidence_metrics(...
        ct_features, radar_features);
end

function volume = combine_volume_measurements(ct_volume, radar_area)
    % Combine volume measurements from both modalities
    % Using weighted average based on confidence
    ct_weight = 0.7;  % CT is typically more accurate for volume
    radar_weight = 0.3;

    % Normalize radar area to volume-like units
    radar_volume = radar_area ^ (3/2);  % Assuming roughly spherical shape

    % Combine measurements
    volume = ct_weight * ct_volume + radar_weight * radar_volume;
end

function surface = combine_surface_metrics(ct_shape, radar_surface)
    % Combine surface measurements from both modalities
    surface = struct();

    % Combine surface area measurements
    surface.area = 0.6 * ct_shape.surface_area + ...
                  0.4 * radar_surface.surface_area;

    % Add shape features from CT
    surface.aspect_ratios = ct_shape.aspect_ratios;

    % Add roughness from radar
    surface.roughness = radar_surface.roughness;
end

function confidence = calculate_confidence_metrics(ct_features, radar_features)
    % Calculate confidence metrics for the fusion
    confidence = struct();

    % Volume measurement confidence
    confidence.volume = assess_volume_confidence(...
        ct_features.volume, ...
        radar_features.surface_metrics.surface_area);

    % Surface measurement confidence
    confidence.surface = assess_surface_confidence(...
        ct_features.shape_metrics.surface_area, ...
        radar_features.surface_metrics.surface_area);

    % Overall confidence score
    confidence.overall = mean([confidence.volume, confidence.surface]);
end

function conf = assess_volume_confidence(ct_vol, radar_area)
    % Assess confidence in volume measurements
    % Based on agreement between modalities
    radar_vol = radar_area ^ (3/2);
    relative_diff = abs(ct_vol - radar_vol) / ct_vol;
    conf = 1 / (1 + relative_diff);
end

function conf = assess_surface_confidence(ct_area, radar_area)
    % Assess confidence in surface measurements
    % Based on agreement between modalities
    relative_diff = abs(ct_area - radar_area) / ct_area;
    conf = 1 / (1 + relative_diff);
end
