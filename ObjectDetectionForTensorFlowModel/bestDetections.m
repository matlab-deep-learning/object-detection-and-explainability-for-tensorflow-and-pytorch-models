function [bboxes,classes,scores,num_box] = bestDetections(inputImage,netOutputMap,scoreThreshold)
%BESTDETECTIONS Get attributes of detections with highest scores
%   This function returns the bounding boxes, classes, scores, and number
%   of boxes for the object detections above the specified threshold.

    detected_bboxes = netOutputMap("detection_boxes");
    ml_bboxes = permute(detected_bboxes.extractdata,[2 3 1]);
    
    detected_scores = netOutputMap("detection_scores");
    ml_scores = detected_scores.extractdata;
    
    detected_classes = netOutputMap("detection_classes");
    ml_classes = detected_classes.extractdata;
       
    det_good_scores_idx = find(ml_scores > scoreThreshold);
    
    scores = ml_scores(det_good_scores_idx);
    classes = ml_classes(det_good_scores_idx);
    det_good_scores_obj_bbox = ml_bboxes(det_good_scores_idx,:);
            
    [im_height, im_width, ~] = size(inputImage);
    T = [im_height im_width im_height im_width];
    det_obj_bbox_un = det_good_scores_obj_bbox .* T;  % ymin xmin ymax xmax
    
    bboxes(:,1) = det_obj_bbox_un(:,2);
    bboxes(:,2) = det_obj_bbox_un(:,1);
    bboxes(:,3) = det_obj_bbox_un(:,4) - det_obj_bbox_un(:,2);
    bboxes(:,4) = det_obj_bbox_un(:,3) - det_obj_bbox_un(:,1);

    num_box = length(scores);
     
end