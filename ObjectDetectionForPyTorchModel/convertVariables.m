function [bboxes,labels,scores] = convertVariables(pt_predictions,img)
% CONVERTVARIABLES Convert Predictions from PyTorch to MATLAB
% This function converts the outputs of the PyTorch model to MATLAB
% variables.

numImg = size(img,4);

if ndims(img)==3
    pt_predictions = struct(pt_predictions);
    bboxes = double(py.numpy.array(pt_predictions.boxes.tolist));
    labels = double(pt_predictions.labels.tolist)';
    scores = double(pt_predictions.scores.tolist)';
    if ~isempty(bboxes)
        bboxes = cat(2,bboxes(:,1:2),bboxes(:,3:4));
        bboxes(:,1:2) = bboxes(:,1:2)+1;
        bboxes(:,3:4) = bboxes(:,3:4)/2;
    end

elseif ndims(img)==4
    a = cell(pt_predictions);
 
    bboxes = cell(numImg,1);
    labels = cell(numImg,1);
    scores = cell(numImg,1);

    for i = 1:numImg
        b = a{i};
        pt_predictions = struct(b);
            
        bboxes{i} = double(py.numpy.array(pt_predictions.boxes.tolist));
        labels{i} = double(pt_predictions.labels.tolist)';
        scores{i} = double(pt_predictions.scores.tolist)';
        if ~isempty(bboxes{i})
            bboxes{i} = cat(2,bboxes{i}(:,1:2),bboxes{i}(:,3:4));
            bboxes{i}(:,1:2) = bboxes{i}(:,1:2)+1;
            bboxes{i}(:,3:4) = bboxes{i}(:,3:4)/2;
        end
    end
    
end