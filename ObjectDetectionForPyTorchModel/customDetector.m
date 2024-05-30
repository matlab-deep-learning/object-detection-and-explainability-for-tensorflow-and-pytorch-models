function [boundingBox,classProb,objectness] = customDetector(img)
%CUSTOMDETECTOR Convert Network to Object Detector
%   This function converts the PyTorch model to a custom object detector
%   that can be used with the D-RISE function.

numImg = size(img,4);

pyrun("from PT_object_detection import loadPTmodel, detectPT")
[model,weights] = pyrun("[a,b] = loadPTmodel()",["a" "b"]);

predictions = pyrun("a = detectPT(b,c,d)","a",b=img,c=model,d=weights);
[boundingBox,labels,objectness] = convertVariables(predictions,img);

if numImg > 1
    classProb = cell(numImg,1);
else
    classProb = [];
end

end
