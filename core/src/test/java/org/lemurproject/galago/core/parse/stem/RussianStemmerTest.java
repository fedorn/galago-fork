/*
 *  BSD License (http://lemurproject.org/galago-license)
 */
package org.lemurproject.galago.core.parse.stem;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 *
 * @author sjh
 */
public class RussianStemmerTest {

  @Test
  public void testLowerCase() {

    Stemmer stemmer = new RussianStemmer();
    assertEquals("кот", stemmer.stemTerm("коты"));
    assertEquals("кот", stemmer.stemTerm("Коты"));

  }
}
