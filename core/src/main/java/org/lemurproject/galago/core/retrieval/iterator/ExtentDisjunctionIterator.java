/*
 *  BSD License (http://www.galagosearch.org/license)
 */
package org.lemurproject.galago.core.retrieval.iterator;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.lemurproject.galago.core.retrieval.query.AnnotatedNode;
import org.lemurproject.galago.core.util.ExtentArray;
import org.lemurproject.galago.tupleflow.Utility;

/**
 *
 * @author sjh
 */
public abstract class ExtentDisjunctionIterator extends DisjunctionIterator implements DataIterator<ExtentArray>, ExtentIterator, CountIterator {

  protected ExtentArray extentCache;

  public ExtentDisjunctionIterator(ExtentIterator[] iterators) throws IOException {
    super(iterators);
    this.extentCache = new ExtentArray();
  }

  @Override
  public boolean hasMatch(long identifier) {
    return super.hasMatch(identifier) && extents().size() > 0;
  }

  @Override
  public String getValueString() throws IOException {
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

  public abstract void loadExtents();

  @Override
  public AnnotatedNode getAnnotatedNode() throws IOException {
    this.loadExtents();
    String type = "extent";
    String className = this.getClass().getSimpleName();
    String parameters = "";
    long document = currentCandidate();
    boolean atCandidate = hasMatch(this.context.document);
    String returnValue = extents().toString();
    List<AnnotatedNode> children = new ArrayList();
    for (BaseIterator child : this.iterators) {
      children.add(child.getAnnotatedNode());
    }
    return new AnnotatedNode(type, className, parameters, document, atCandidate, returnValue, children);
  }
}
