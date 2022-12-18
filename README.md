# onScanner
**The used algorithm is Douglas–Peucker algorithm**

In our case, we used this algorithm to get only 4 corner points among many points. By getting 4 corner points we could draw a line between these points to outline the 
document.
Contour approximation, which uses the **Ramer**–**Douglas**–**Peucker (RDP)**
 algorithm, aims to simplify a polyline by reducing its vertices given a threshold value. In layman terms, we take a curve and reduce its number of vertices while retaining the bulk of its shape.

Given the start and end points of a curve, the algorithm will first find the vertex at maximum distance from the line joining the two reference points. Let’s refer to it as max_point. If the max_point lies at a distance less than the threshold, we automatically neglect all the vertices between the start and end points and make the curve a straight line.

It removes points that contribute very little (epsilon) to the shape of the contour. Colinear points are a trivial case because they contribute *zero* to the shape of the contour. The most prominent corners are left standing. The result is an approximation of the input contour.

