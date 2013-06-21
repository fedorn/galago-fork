package org.lemurproject.galago.core.index.source;

import java.io.IOException;
import org.lemurproject.galago.core.index.BTreeReader;
import org.lemurproject.galago.tupleflow.Utility;

/**
 *
 * @author jfoley
 */
public abstract class BTreeValueSource implements DiskSource {
  // OPTIONS
  public static final int HAS_SKIPS = 0x01;
  public static final int HAS_MAXTF = 0x02;
  public static final int HAS_INLINING = 0x04;
  
  final protected BTreeReader.BTreeIterator btreeIter;
  final protected byte[] key;
  
  public BTreeValueSource(BTreeReader.BTreeIterator it) throws IOException {
    this.key = it.getKey();
    btreeIter = it;
  }
  
  @Override
  public boolean hasMatch(long id) {
    return !isDone() && currentCandidate() == id;
  }
  
  @Override
  public String key() {
    return Utility.toString(key);
  }
}