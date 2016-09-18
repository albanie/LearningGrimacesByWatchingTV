function img_names = get_pngs_in_subdirs(path)
% returns an nx1 cell array of paths to any .png files
% found recursively in subdirectories of 'path'

% create a cell array of all subdirectories
subdirs = strsplit(genpath(path), ':')';

% extract the jpgs in each of these subdirectories
pngs = cellfun(@get_pngs_in_dir, subdirs, 'UniformOutput', false);

% flatten the cell array
img_names = vertcat(pngs{:});
end

% -----------------------------------------
function img_names = get_pngs_in_dir(path)
% -----------------------------------------
% List all files in the given directory with .png extension
img_files = dir(fullfile(path, '*.png'));

% create a cell array of the full paths to each image
img_names = {img_files.name}';

% ignore OS created backup files
img_names = img_names(find(~strncmpi(img_names, '._', 2)));

paths = repmat({path}, size(img_names));
img_names = strcat(paths, filesep, img_names);

end
