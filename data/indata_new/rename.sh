for file in *areaOfInterest*.*; do mv "$file" "${file/*areaOfInterest*./areaOfInterest.}"; done
for file in *hydrographyL*.*; do mv "$file" "${file/*hydrographyL*./hydrographyL.}"; done
for file in *hydrographyA*.*; do mv "$file" "${file/*hydrographyA*./hydrographyA.}"; done
for file in *observedEvent*.*; do mv "$file" "${file/*observedEventA*./flood.}"; done
