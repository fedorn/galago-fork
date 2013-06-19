/*
 *  BSD License (http://www.galagosearch.org/license)
 */
package org.lemurproject.galago.core.retrieval.iterator;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.lemurproject.galago.core.retrieval.query.AnnotatedNode;
import org.lemurproject.galago.core.retrieval.query.NodeParameters;
import org.lemurproject.galago.core.util.ExtentArray;
import org.lemurproject.galago.tupleflow.Utility;

/**
 *
 * @author sjh
 */
public abstract class ExtentConjunctionIterator extends ConjunctionIterator implements MovableDataIterator<ExtentArray>, ExtentIterator, CountIterator {

  protected ExtentArray extentCache;
  protected byte[] key;

  public ExtentConjunctionIterator(NodeParameters parameters, ExtentIterator[] iterators) throws IOException {
    super(parameters, iterators);
    this.extentCache = new ExtentArray();
    buildKey(iterators);
  }

  @Override
  public boolean hasMatch(int identifier) {
    return super.hasMatch(identifier) && extents().size() > 0;
  }

  @Override
  public String getEntry() throws IOException {
    ArrayList<String> strs = new ArrayList<String>();
    ExtentArrayIterator eai = new ExtentArrayIterator(extents());
    while (!eai.isDone()) {
      strs.add(String.format("[%d, %d]", eai.currentBegin(), eai.currentEnd()));
      eai.next();
    }
    return Utility.join(strs.toArray(new String[0]), ",");
  }

  @Override
  public ExtentArray extents() {
    this.loadExtents();
    return extentCache;
  }

  @Override
  public ExtentArray getData() {
    return extents();
  }

  @Override
  public int count() {
    return extents().size();
  }

  @Override
  public int maximumCount() {
    int min = 0;
    for (int i = 0; i < iterators.length; i++) {
      min = Math.min(min, ((CountIterator) iterators[i]).maximumCount());
    }
    return min;
  }

  public abstract void loadExtents();

  @Override
  public AnnotatedNode getAnnotatedNode() throws IOException {
    // ensure extentCache are loaded
    this.loadExtents();

    String type = "extent";
    String className = this.getClass().getSimpleName();
    String parameters = "";
    int document = currentCandidate();
    boolean atCandidate = hasMatch(this.context.document);
    String returnValue = extents().toString();
    List<AnnotatedNode> children = new ArrayList();
    for (MovableIterator child : this.iterators) {
      children.add(child.getAnnotatedNode());
    }
    return new AnnotatedNode(type, className, parameters, document, atCandidate, returnValue, children);
  }

  @Override
  public byte[] key() {
    return key;
  }

  protected void buildKey(ExtentIterator[] iterators) {
    int keysize = 2;
    for (int i = 0; i < iterators.length; i++) {
      keysize += iterators[i].key().length;
    }
    key = new byte[keysize];
    keysize = 2;
    // conjunction marker                                                                                                                   
    key[0] = 'C' >> 8; // conjunction marker;                                                                                               
    key[1] = 'C' & 0xFF;
    for (int i = 0; i < iterators.length; i++) {
      ExtentIterator it = iterators[i];
      byte[] inner = it.key();
      System.arraycopy(inner, 0, key, keysize, inner.length);
      keysize += inner.length;
    }
  }
}
