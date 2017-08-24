/*
 *  BSD License (http://www.galagosearch.org/license)
 */
package org.lemurproject.galago.core.eval.metric;

/**
 * <p>Normalized Discounted Cumulative Gain (linear formulation)</p>
 *
 * We use same formula as trec_eval:
 *
 * Score = N \sum_i r(i) / \log(1 + i)
 *
 */
public class LinearNormalizedDiscountedCumulativeGain extends NormalizedDiscountedCumulativeGain {

  public LinearNormalizedDiscountedCumulativeGain(String metric, int documentsRetrieved) {
    super(metric, documentsRetrieved);
  }

  public LinearNormalizedDiscountedCumulativeGain(String metric) {
    super(metric);
  }

  public LinearNormalizedDiscountedCumulativeGain(int documentsRetrieved) {
    super("linndcg" + documentsRetrieved, documentsRetrieved);
  }

  /**
   * Computes dcg @ documentsRetrieved
   *  
   */
  @Override
  protected double computeDCG(double[] gains) {
    double dcg = 0.0;
    for (int i = 0; i < Math.min(gains.length, this.documentsRetrieved); i++) {
      dcg += gains[i] / Math.log(i + 2);
    }
    return dcg;
  }
}
