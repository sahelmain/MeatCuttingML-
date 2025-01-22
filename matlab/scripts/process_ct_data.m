function [features] = process_ct_data(ct_image_path)
    % Process CT scan data and extract relevant features
    %
    % Args:
    %   ct_image_path: Path to the CT image file
    %
    % Returns:
    %   features: Extracted features from CT scan

    % Load CT image
    ct_image = dicomread(ct_image_path);

    % Pre-processing
    ct_image = double(ct_image);
    ct_image = normalize_hounsfield_units(ct_image);

    % Segment different tissue types
    [muscle, fat, bone] = segment_tissues(ct_image);

    % Extract features
    features = struct();
    features.volume = calculate_volume(muscle);
    features.density = calculate_density(ct_image);
    features.tissue_ratios = calculate_tissue_ratios(muscle, fat, bone);
    features.shape_metrics = extract_shape_features(muscle);
end

function normalized = normalize_hounsfield_units(image)
    % Convert to Hounsfield units and normalize
    min_hu = -1000;
    max_hu = 1000;
    normalized = (image - min_hu) / (max_hu - min_hu);
end

function [muscle, fat, bone] = segment_tissues(image)
    % Segment different tissue types based on HU ranges
    muscle = (image > 30) & (image < 100);
    fat = (image > -100) & (image < -50);
    bone = image > 200;
end

function volume = calculate_volume(mask)
    % Calculate volume from binary mask
    volume = sum(mask(:)) * voxel_size();
end

function density = calculate_density(image)
    % Calculate tissue density metrics
    density = mean(image(image > -50 & image < 100));
end

function ratios = calculate_tissue_ratios(muscle, fat, bone)
    % Calculate ratios between different tissue types
    total = sum(muscle(:)) + sum(fat(:)) + sum(bone(:));
    ratios = struct();
    ratios.muscle = sum(muscle(:)) / total;
    ratios.fat = sum(fat(:)) / total;
    ratios.bone = sum(bone(:)) / total;
end

function metrics = extract_shape_features(mask)
    % Extract shape-based features
    props = regionprops3(mask, 'Volume', 'SurfaceArea', 'PrincipalAxisLength');
    metrics = struct();
    metrics.volume = props.Volume;
    metrics.surface_area = props.SurfaceArea;
    metrics.aspect_ratios = props.PrincipalAxisLength(1,:);
end

function size = voxel_size()
    % Return voxel size in mm^3
    size = 1.0; % Default value, should be updated based on CT scanner specs
end
