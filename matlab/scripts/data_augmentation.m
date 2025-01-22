function [augmented_data] = data_augmentation(ct_data, radar_data, num_variations)
    % Generate augmented versions of CT and radar data
    %
    % Args:
    %   ct_data: Original CT scan data
    %   radar_data: Original radar measurements
    %   num_variations: Number of augmented samples to generate
    %
    % Returns:
    %   augmented_data: Structure containing augmented datasets

    augmented_data = struct();

    % CT data augmentation
    augmented_data.ct = augment_ct_data(ct_data, num_variations);

    % Radar data augmentation
    augmented_data.radar = augment_radar_data(radar_data, num_variations);

    % Ensure consistency between modalities
    augmented_data = ensure_consistency(augmented_data);
end

function augmented_ct = augment_ct_data(ct_data, num_variations)
    % Generate variations of CT data
    augmented_ct = cell(num_variations, 1);

    for i = 1:num_variations
        % Apply random transformations
        transformed = apply_ct_transformations(ct_data);

        % Add realistic noise
        noisy = add_ct_noise(transformed);

        % Adjust tissue distributions
        augmented = vary_tissue_distribution(noisy);

        augmented_ct{i} = augmented;
    end
end

function transformed = apply_ct_transformations(image)
    % Apply geometric transformations

    % Random rotation
    angle = rand() * 20 - 10;  % -10 to +10 degrees
    transformed = imrotate(image, angle, 'bilinear', 'crop');

    % Random scaling
    scale = 0.9 + rand() * 0.2;  % 0.9 to 1.1
    transformed = imresize(transformed, scale);

    % Random translation
    transformed = random_translation(transformed);
end

function noisy = add_ct_noise(image)
    % Add realistic CT noise

    % Gaussian noise
    noise_level = 0.02 * rand();
    gaussian_noise = noise_level * randn(size(image));

    % Poisson noise (quantum noise)
    poisson_noise = poissrnd(image) - image;

    % Combine noise types
    noisy = image + gaussian_noise + 0.5 * poisson_noise;
end

function varied = vary_tissue_distribution(image)
    % Vary tissue distributions while maintaining realism

    % Adjust tissue contrasts
    contrast_scale = 0.9 + rand() * 0.2;
    varied = image * contrast_scale;

    % Modify tissue boundaries
    varied = modify_tissue_boundaries(varied);
end

function augmented_radar = augment_radar_data(radar_data, num_variations)
    % Generate variations of radar data
    augmented_radar = cell(num_variations, 1);

    for i = 1:num_variations
        % Apply signal transformations
        transformed = apply_radar_transformations(radar_data);

        % Add realistic noise
        noisy = add_radar_noise(transformed);

        % Vary surface characteristics
        augmented = vary_surface_characteristics(noisy);

        augmented_radar{i} = augmented;
    end
end

function transformed = apply_radar_transformations(signal)
    % Apply radar signal transformations

    % Time scaling
    scale = 0.95 + rand() * 0.1;  % 0.95 to 1.05
    transformed = resample(signal, round(length(signal) * scale), length(signal));

    % Phase shifts
    phase_shift = rand() * 2 * pi;
    transformed = transformed * exp(1i * phase_shift);
end

function noisy = add_radar_noise(signal)
    % Add realistic radar noise

    % Thermal noise
    noise_power = 0.01 * rand();
    thermal_noise = sqrt(noise_power/2) * (randn(size(signal)) + 1i * randn(size(signal)));

    % Phase noise
    phase_noise = add_phase_noise(signal);

    % Combine noise types
    noisy = signal + thermal_noise + phase_noise;
end

function varied = vary_surface_characteristics(signal)
    % Vary surface reflection characteristics

    % Modify reflection coefficients
    refl_scale = 0.9 + rand() * 0.2;
    varied = signal * refl_scale;

    % Add surface roughness variations
    varied = add_roughness_variations(varied);
end

function consistent_data = ensure_consistency(augmented_data)
    % Ensure consistency between CT and radar augmentations

    for i = 1:length(augmented_data.ct)
        % Align geometric transformations
        [augmented_data.ct{i}, augmented_data.radar{i}] = ...
            align_transformations(augmented_data.ct{i}, augmented_data.radar{i});

        % Ensure physical consistency
        [augmented_data.ct{i}, augmented_data.radar{i}] = ...
            enforce_physical_constraints(augmented_data.ct{i}, augmented_data.radar{i});
    end

    consistent_data = augmented_data;
end

% Helper functions
function translated = random_translation(image)
    max_shift = round(size(image, 1) * 0.1);
    shift_x = randi([-max_shift, max_shift]);
    shift_y = randi([-max_shift, max_shift]);
    translated = imtranslate(image, [shift_x, shift_y]);
end

function modified = modify_tissue_boundaries(image)
    % Apply morphological operations to modify boundaries
    se = strel('disk', 2);
    if rand() > 0.5
        modified = imdilate(image, se);
    else
        modified = imerode(image, se);
    end
end

function noisy = add_phase_noise(signal)
    % Add phase noise to radar signal
    phase_noise_power = 0.005 * rand();
    phase_noise = exp(1i * sqrt(phase_noise_power) * randn(size(signal)));
    noisy = signal .* phase_noise;
end

function varied = add_roughness_variations(signal)
    % Add variations in surface roughness
    roughness_scale = 0.8 + rand() * 0.4;
    varied = signal + roughness_scale * randn(size(signal));
end

function [ct_aligned, radar_aligned] = align_transformations(ct, radar)
    % Align geometric transformations between modalities
    % This is a placeholder for more sophisticated alignment
    ct_aligned = ct;
    radar_aligned = radar;
end

function [ct_consistent, radar_consistent] = enforce_physical_constraints(ct, radar)
    % Enforce physical constraints between modalities
    % This is a placeholder for more sophisticated constraints
    ct_consistent = ct;
    radar_consistent = radar;
end
