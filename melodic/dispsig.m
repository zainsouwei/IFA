function dispsig(signalMatrix);




numSignals = size (signalMatrix, 2);

for i = 1:numSignals,
  subplot (numSignals,1, i);
  plot (signalMatrix (:,i));
end
% Move the handle to the first subplot, so that the title may
% easily be added to the top.
subplot (numSignals,1, 1);
