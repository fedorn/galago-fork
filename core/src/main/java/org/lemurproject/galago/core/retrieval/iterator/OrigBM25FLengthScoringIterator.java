// BSD License (http://lemurproject.org/galago-license)
package org.lemurproject.galago.core.retrieval.iterator;

import org.lemurproject.galago.core.retrieval.processing.ScoringContext;
import org.lemurproject.galago.core.retrieval.query.NodeParameters;

import java.io.IOException;

public class OrigBM25FLengthScoringIterator extends ScoringFunctionIterator {

  public OrigBM25FLengthScoringIterator(NodeParameters p, LengthsIterator ls, CountIterator it)
          throws IOException {
    super(p, ls, it);
  }

  @Override
  public double score(ScoringContext c) {
    return lengthsIterator.length(c);
  }

}
