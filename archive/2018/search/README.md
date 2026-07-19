# Search

Papers regarding topics of path finding, tree spanning, etc.

## Overview 

- [R* Search (Likhachev, Stentz 2008)](RStarSearch.pdf): Heuristic search which depends less on the quality of the hueristic function
- [D* Search (Koenig, Likhachev 2005)](FastReplanningForNavigationInUnknownTerrain.pdf): Heuristic search method that repeatedly determines a shortest path while exploring
- [Anytime RRTs (Ferguson, Stentz 2006)](AnytimeRRTs.pdf): Anytime algorithm for planning paths generating RRTs
- [A* Search (Hart, Nilsson, Raphael 1968)](AStarSearch.pdf): Best-first search algorithm which searches amongst all possible paths that incur the smallest cost and that appear to lead most quickly to the goal node

## Next Steps

- Implement [Dijkstra's Algorithm](https://en.wikipedia.org/wiki/Dijkstra's_algorithm) and then [A* Search Algorithm](https://en.wikipedia.org/wiki/A*_search_algorithm)

### Further Readings

- [CCG Parsing (Lee, Lewis, Zettlemoyer 2016)](CCGParsing.pdf)
- [Jump Point Search (Harabor, Grastien 2011)](OnlineGraphPruningForPathfindingOnGridMaps.pdf)
- [A* Parsing (Klein, Manning 2003)](AStarParsing.pdf)

## Thoughts and Reflection

### Inspiration from [Dijkstra's Algorithm](https://en.wikipedia.org/wiki/Dijkstra's_algorithm)

Overall it seems as though one with some familiarity and inspiration from [Dijkstra's Algorithm](https://en.wikipedia.org/wiki/Dijkstra's_algorithm), see the improvement possible via heuristics; just as Hart, Nilsson and Raphael proved. Further improvements can thus be obtained via optimizations in different environments (D* for exploring with incomplete information, R* for more complex problems and less reliance on the heuristic).

### [Anytime Algorithm](https://en.wikipedia.org/wiki/Anytime_algorithm)

With the concept of an Anytime Algorithm, it seems incredibly practical to create an algorithm which is guaranteed to produce better results given more resources (namely, memory and compute time), and even better yet, can return results incremetally. 
