function [boundingBox,classProb,objectness] = customDetector(img)
%CUSTOMDETECTOR Convert Network to Object Detector
%   This function converts the network imported from TensorFlow to a custom
%   object detector that can be used with the D-RISE function.

persistent detector;
if isempty(detector)
    detector = coder.loadDeepLearningNetwork("centernet_object_detector.mat");
end

numImg = size(img,4);

if numImg >1
    boundingBox = cell(numImg,1);
    classProb = cell(numImg,1);
    objectness = cell(numImg,1);

    for j = 1:numImg
        [y1,y2,y3,y4] = detector.predict(dlarray(single(img(:,:,:,j)),"SSCB"));

        % Create map
        netOutputNames = detector.OutputNames';
        netOutputMap = containers.Map;
        netOutputs = {y1,y2,y3,y4};
        for i = 1:numel(netOutputNames)
            opNameStrSplit = strsplit(netOutputNames{i},'/');
            opName = opNameStrSplit{end};
            netOutputMap(opName) = netOutputs{i};
        end

        % Get bounding boxes and scores - only the best
        [bboxes,~,scores] = bestDetections(img,netOutputMap,0.6);

        boundingBox{j} = bboxes;
        objectness{j} = scores';
    end

else
    [y1,y2,y3,y4] = detector.predict(dlarray(single(img),"SSCB"));

    % Create map
    netOutputNames = detector.OutputNames';
    netOutputMap = containers.Map;
    netOutputs = {y1,y2,y3,y4};
    for i = 1:numel(netOutputNames)
        opNameStrSplit = strsplit(netOutputNames{i},'/');
        opName = opNameStrSplit{end};
        netOutputMap(opName) = netOutputs{i};
    end

    % Get bounding boxes and scores - only the best
    [bboxes,~,scores] = bestDetections(img,netOutputMap,0.6);

    boundingBox = bboxes;
    objectness = scores';
    classProb = [];
end

end
