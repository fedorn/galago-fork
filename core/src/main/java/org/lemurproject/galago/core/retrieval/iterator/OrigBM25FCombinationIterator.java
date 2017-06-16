// BSD License (http://www.galagosearch.org/license)
package org.lemurproject.galago.core.retrieval.iterator;

import org.lemurproject.galago.core.retrieval.RequiredParameters;
import org.lemurproject.galago.core.retrieval.RequiredStatistics;
import org.lemurproject.galago.core.retrieval.processing.ScoringContext;
import org.lemurproject.galago.core.retrieval.query.NodeParameters;

public class OrigBM25FCombinationIterator extends ScoreCombinationIterator {

  double[] idfs;


  public OrigBM25FCombinationIterator(NodeParameters parameters,
                                      ScoreIterator[] childIterators) {
    super(parameters, childIterators);
    idfs = new double[iterators.length];
    for (int i = 0; i < idfs.length; i++) {
      idfs[i] = parameters.getDouble("idf" + i);
    }
  }

  @Override
  public double score(ScoringContext c) {
    double total = 0;
    double length = scoreIterators[0].score(c);
    for (int i = 0; i < scoreIterators.length; i++) {
      double score = scoreIterators[i].score(c);
      // the second iterator here is the idf iterator - well, it should be
      total += weights[i] * score * idfs[i];
    }
    return  total;
  }

  @Override
  public double minimumScore() {
    double min = 0;
    double score;
    for (int i = 0; i < scoreIterators.length; i++) {
      score = scoreIterators[i].minimumScore();
      min += weights[i] * score * idfs[i];
    }
    return min;
  }

  @Override
  public double maximumScore() {
    double max = 0;
    double score;
    for (int i = 0; i < scoreIterators.length; i++) {
      score = scoreIterators[i].maximumScore();
      max += weights[i] * score * idfs[i];
    }
    return max;
  }
}
