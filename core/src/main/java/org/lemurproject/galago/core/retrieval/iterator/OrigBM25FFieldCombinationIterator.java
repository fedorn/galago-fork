// BSD License (http://www.galagosearch.org/license)
package org.lemurproject.galago.core.retrieval.iterator;

import org.lemurproject.galago.core.retrieval.RequiredParameters;
import org.lemurproject.galago.core.retrieval.RequiredStatistics;
import org.lemurproject.galago.core.retrieval.processing.ScoringContext;
import org.lemurproject.galago.core.retrieval.query.NodeParameters;

@RequiredStatistics(statistics = {"collectionLength", "documentCount"})
@RequiredParameters(parameters = {"b", "K"})
public class OrigBM25FFieldCombinationIterator extends ScoreCombinationIterator {

  double K = -1;
  double b = -1;
  double avgDocLength;

  public OrigBM25FFieldCombinationIterator(NodeParameters parameters,
                                           ScoreIterator[] childIterators) {
    super(parameters, childIterators);
    K = parameters.getDouble("K");
    b = parameters.getDouble("b");
    long collectionLength = parameters.getLong("collectionLength");
    long documentCount = parameters.getLong("documentCount");
    avgDocLength = (collectionLength + 0.0) / (documentCount + 0.0);

  }

  @Override
  public double score(ScoringContext c) {
    double total = 0;
    for (int i = 0; i < scoreIterators.length - 1; i++) {
      double score = scoreIterators[i].score(c);
      total += weights[i] * score;
    }
    double length = scoreIterators[scoreIterators.length - 1].score(c);
    double numerator = total * (K + 1);
    double denominator = total + (K * (1 - b + (b * length / avgDocLength)));
    return numerator / denominator;
  }

  @Override
  public double minimumScore() {
    throw new java.lang.UnsupportedOperationException("Not supported.");
  }

  @Override
  public double maximumScore() {
    throw new java.lang.UnsupportedOperationException("Not supported.");
  }
}
