// BSD License (http://www.galagosearch.org/license)
package org.lemurproject.galago.core.retrieval.iterator;

import org.lemurproject.galago.core.retrieval.RequiredParameters;
import org.lemurproject.galago.core.retrieval.RequiredStatistics;
import org.lemurproject.galago.core.retrieval.processing.ScoringContext;
import org.lemurproject.galago.core.retrieval.query.NodeParameters;

@RequiredParameters(parameters = {"background", "mu"})
public class TranslationModelCombinationIterator extends ScoreCombinationIterator {

  double background;
  double mu;

  public TranslationModelCombinationIterator(NodeParameters parameters,
                                             ScoreIterator[] childIterators) {
    super(parameters, childIterators);
    background = parameters.getDouble("background");
    mu = parameters.get("mu", 1500D);
  }

  @Override
  public double score(ScoringContext c) {
    double tf_hat = 0;
    for (int i = 0; i < scoreIterators.length - 1; i++) {
      double score = scoreIterators[i].score(c);
      tf_hat += weights[i] * score;
    }
    double length = scoreIterators[scoreIterators.length - 1].score(c);
    double numerator = tf_hat + mu * background;
    double denominator = length + mu;
    return Math.log(numerator / denominator);
  }

  @Override
  public double minimumScore() {
    throw new UnsupportedOperationException("Not supported.");
  }

  @Override
  public double maximumScore() {
    throw new UnsupportedOperationException("Not supported.");
  }
}
