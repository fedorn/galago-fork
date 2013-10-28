// BSD License (http://lemurproject.org/galago-license)
package org.lemurproject.galago.tupleflow;

import java.io.IOException;
import org.lemurproject.galago.utility.Parameters;

/**
 *
 * @author trevor
 */
public interface TupleFlowParameters {
    public Parameters getJSON();
    public TypeReader getTypeReader(String specification) throws IOException;
    public Processor getTypeWriter(String specification) throws IOException;
    public Counter getCounter(String name);

    public boolean readerExists(String specification, String className, String[] order);
    public boolean writerExists(String specification, String className, String[] order);

    public int getInstanceId();
}
