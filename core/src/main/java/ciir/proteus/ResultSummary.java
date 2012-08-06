/**
 * Autogenerated by Thrift Compiler (0.8.0)
 *
 * DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
 *  @generated
 */
package ciir.proteus;

import org.apache.thrift.scheme.IScheme;
import org.apache.thrift.scheme.SchemeFactory;
import org.apache.thrift.scheme.StandardScheme;

import org.apache.thrift.scheme.TupleScheme;
import org.apache.thrift.protocol.TTupleProtocol;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.util.EnumMap;
import java.util.Set;
import java.util.HashSet;
import java.util.EnumSet;
import java.util.Collections;
import java.util.BitSet;
import java.nio.ByteBuffer;
import java.util.Arrays;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ResultSummary implements org.apache.thrift.TBase<ResultSummary, ResultSummary._Fields>, java.io.Serializable, Cloneable {
  private static final org.apache.thrift.protocol.TStruct STRUCT_DESC = new org.apache.thrift.protocol.TStruct("ResultSummary");

  private static final org.apache.thrift.protocol.TField TEXT_FIELD_DESC = new org.apache.thrift.protocol.TField("text", org.apache.thrift.protocol.TType.STRING, (short)1);
  private static final org.apache.thrift.protocol.TField HIGHLIGHTS_FIELD_DESC = new org.apache.thrift.protocol.TField("highlights", org.apache.thrift.protocol.TType.LIST, (short)2);

  private static final Map<Class<? extends IScheme>, SchemeFactory> schemes = new HashMap<Class<? extends IScheme>, SchemeFactory>();
  static {
    schemes.put(StandardScheme.class, new ResultSummaryStandardSchemeFactory());
    schemes.put(TupleScheme.class, new ResultSummaryTupleSchemeFactory());
  }

  public String text; // required
  public List<TextRegion> highlights; // required

  /** The set of fields this struct contains, along with convenience methods for finding and manipulating them. */
  public enum _Fields implements org.apache.thrift.TFieldIdEnum {
    TEXT((short)1, "text"),
    HIGHLIGHTS((short)2, "highlights");

    private static final Map<String, _Fields> byName = new HashMap<String, _Fields>();

    static {
      for (_Fields field : EnumSet.allOf(_Fields.class)) {
        byName.put(field.getFieldName(), field);
      }
    }

    /**
     * Find the _Fields constant that matches fieldId, or null if its not found.
     */
    public static _Fields findByThriftId(int fieldId) {
      switch(fieldId) {
        case 1: // TEXT
          return TEXT;
        case 2: // HIGHLIGHTS
          return HIGHLIGHTS;
        default:
          return null;
      }
    }

    /**
     * Find the _Fields constant that matches fieldId, throwing an exception
     * if it is not found.
     */
    public static _Fields findByThriftIdOrThrow(int fieldId) {
      _Fields fields = findByThriftId(fieldId);
      if (fields == null) throw new IllegalArgumentException("Field " + fieldId + " doesn't exist!");
      return fields;
    }

    /**
     * Find the _Fields constant that matches name, or null if its not found.
     */
    public static _Fields findByName(String name) {
      return byName.get(name);
    }

    private final short _thriftId;
    private final String _fieldName;

    _Fields(short thriftId, String fieldName) {
      _thriftId = thriftId;
      _fieldName = fieldName;
    }

    public short getThriftFieldId() {
      return _thriftId;
    }

    public String getFieldName() {
      return _fieldName;
    }
  }

  // isset id assignments
  public static final Map<_Fields, org.apache.thrift.meta_data.FieldMetaData> metaDataMap;
  static {
    Map<_Fields, org.apache.thrift.meta_data.FieldMetaData> tmpMap = new EnumMap<_Fields, org.apache.thrift.meta_data.FieldMetaData>(_Fields.class);
    tmpMap.put(_Fields.TEXT, new org.apache.thrift.meta_data.FieldMetaData("text", org.apache.thrift.TFieldRequirementType.DEFAULT, 
        new org.apache.thrift.meta_data.FieldValueMetaData(org.apache.thrift.protocol.TType.STRING)));
    tmpMap.put(_Fields.HIGHLIGHTS, new org.apache.thrift.meta_data.FieldMetaData("highlights", org.apache.thrift.TFieldRequirementType.DEFAULT, 
        new org.apache.thrift.meta_data.ListMetaData(org.apache.thrift.protocol.TType.LIST, 
            new org.apache.thrift.meta_data.StructMetaData(org.apache.thrift.protocol.TType.STRUCT, TextRegion.class))));
    metaDataMap = Collections.unmodifiableMap(tmpMap);
    org.apache.thrift.meta_data.FieldMetaData.addStructMetaDataMap(ResultSummary.class, metaDataMap);
  }

  public ResultSummary() {
  }

  public ResultSummary(
    String text,
    List<TextRegion> highlights)
  {
    this();
    this.text = text;
    this.highlights = highlights;
  }

  /**
   * Performs a deep copy on <i>other</i>.
   */
  public ResultSummary(ResultSummary other) {
    if (other.isSetText()) {
      this.text = other.text;
    }
    if (other.isSetHighlights()) {
      List<TextRegion> __this__highlights = new ArrayList<TextRegion>();
      for (TextRegion other_element : other.highlights) {
        __this__highlights.add(new TextRegion(other_element));
      }
      this.highlights = __this__highlights;
    }
  }

  public ResultSummary deepCopy() {
    return new ResultSummary(this);
  }

  @Override
  public void clear() {
    this.text = null;
    this.highlights = null;
  }

  public String getText() {
    return this.text;
  }

  public ResultSummary setText(String text) {
    this.text = text;
    return this;
  }

  public void unsetText() {
    this.text = null;
  }

  /** Returns true if field text is set (has been assigned a value) and false otherwise */
  public boolean isSetText() {
    return this.text != null;
  }

  public void setTextIsSet(boolean value) {
    if (!value) {
      this.text = null;
    }
  }

  public int getHighlightsSize() {
    return (this.highlights == null) ? 0 : this.highlights.size();
  }

  public java.util.Iterator<TextRegion> getHighlightsIterator() {
    return (this.highlights == null) ? null : this.highlights.iterator();
  }

  public void addToHighlights(TextRegion elem) {
    if (this.highlights == null) {
      this.highlights = new ArrayList<TextRegion>();
    }
    this.highlights.add(elem);
  }

  public List<TextRegion> getHighlights() {
    return this.highlights;
  }

  public ResultSummary setHighlights(List<TextRegion> highlights) {
    this.highlights = highlights;
    return this;
  }

  public void unsetHighlights() {
    this.highlights = null;
  }

  /** Returns true if field highlights is set (has been assigned a value) and false otherwise */
  public boolean isSetHighlights() {
    return this.highlights != null;
  }

  public void setHighlightsIsSet(boolean value) {
    if (!value) {
      this.highlights = null;
    }
  }

  public void setFieldValue(_Fields field, Object value) {
    switch (field) {
    case TEXT:
      if (value == null) {
        unsetText();
      } else {
        setText((String)value);
      }
      break;

    case HIGHLIGHTS:
      if (value == null) {
        unsetHighlights();
      } else {
        setHighlights((List<TextRegion>)value);
      }
      break;

    }
  }

  public Object getFieldValue(_Fields field) {
    switch (field) {
    case TEXT:
      return getText();

    case HIGHLIGHTS:
      return getHighlights();

    }
    throw new IllegalStateException();
  }

  /** Returns true if field corresponding to fieldID is set (has been assigned a value) and false otherwise */
  public boolean isSet(_Fields field) {
    if (field == null) {
      throw new IllegalArgumentException();
    }

    switch (field) {
    case TEXT:
      return isSetText();
    case HIGHLIGHTS:
      return isSetHighlights();
    }
    throw new IllegalStateException();
  }

  @Override
  public boolean equals(Object that) {
    if (that == null)
      return false;
    if (that instanceof ResultSummary)
      return this.equals((ResultSummary)that);
    return false;
  }

  public boolean equals(ResultSummary that) {
    if (that == null)
      return false;

    boolean this_present_text = true && this.isSetText();
    boolean that_present_text = true && that.isSetText();
    if (this_present_text || that_present_text) {
      if (!(this_present_text && that_present_text))
        return false;
      if (!this.text.equals(that.text))
        return false;
    }

    boolean this_present_highlights = true && this.isSetHighlights();
    boolean that_present_highlights = true && that.isSetHighlights();
    if (this_present_highlights || that_present_highlights) {
      if (!(this_present_highlights && that_present_highlights))
        return false;
      if (!this.highlights.equals(that.highlights))
        return false;
    }

    return true;
  }

  @Override
  public int hashCode() {
    return 0;
  }

  public int compareTo(ResultSummary other) {
    if (!getClass().equals(other.getClass())) {
      return getClass().getName().compareTo(other.getClass().getName());
    }

    int lastComparison = 0;
    ResultSummary typedOther = (ResultSummary)other;

    lastComparison = Boolean.valueOf(isSetText()).compareTo(typedOther.isSetText());
    if (lastComparison != 0) {
      return lastComparison;
    }
    if (isSetText()) {
      lastComparison = org.apache.thrift.TBaseHelper.compareTo(this.text, typedOther.text);
      if (lastComparison != 0) {
        return lastComparison;
      }
    }
    lastComparison = Boolean.valueOf(isSetHighlights()).compareTo(typedOther.isSetHighlights());
    if (lastComparison != 0) {
      return lastComparison;
    }
    if (isSetHighlights()) {
      lastComparison = org.apache.thrift.TBaseHelper.compareTo(this.highlights, typedOther.highlights);
      if (lastComparison != 0) {
        return lastComparison;
      }
    }
    return 0;
  }

  public _Fields fieldForId(int fieldId) {
    return _Fields.findByThriftId(fieldId);
  }

  public void read(org.apache.thrift.protocol.TProtocol iprot) throws org.apache.thrift.TException {
    schemes.get(iprot.getScheme()).getScheme().read(iprot, this);
  }

  public void write(org.apache.thrift.protocol.TProtocol oprot) throws org.apache.thrift.TException {
    schemes.get(oprot.getScheme()).getScheme().write(oprot, this);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder("ResultSummary(");
    boolean first = true;

    sb.append("text:");
    if (this.text == null) {
      sb.append("null");
    } else {
      sb.append(this.text);
    }
    first = false;
    if (!first) sb.append(", ");
    sb.append("highlights:");
    if (this.highlights == null) {
      sb.append("null");
    } else {
      sb.append(this.highlights);
    }
    first = false;
    sb.append(")");
    return sb.toString();
  }

  public void validate() throws org.apache.thrift.TException {
    // check for required fields
  }

  private void writeObject(java.io.ObjectOutputStream out) throws java.io.IOException {
    try {
      write(new org.apache.thrift.protocol.TCompactProtocol(new org.apache.thrift.transport.TIOStreamTransport(out)));
    } catch (org.apache.thrift.TException te) {
      throw new java.io.IOException(te);
    }
  }

  private void readObject(java.io.ObjectInputStream in) throws java.io.IOException, ClassNotFoundException {
    try {
      read(new org.apache.thrift.protocol.TCompactProtocol(new org.apache.thrift.transport.TIOStreamTransport(in)));
    } catch (org.apache.thrift.TException te) {
      throw new java.io.IOException(te);
    }
  }

  private static class ResultSummaryStandardSchemeFactory implements SchemeFactory {
    public ResultSummaryStandardScheme getScheme() {
      return new ResultSummaryStandardScheme();
    }
  }

  private static class ResultSummaryStandardScheme extends StandardScheme<ResultSummary> {

    public void read(org.apache.thrift.protocol.TProtocol iprot, ResultSummary struct) throws org.apache.thrift.TException {
      org.apache.thrift.protocol.TField schemeField;
      iprot.readStructBegin();
      while (true)
      {
        schemeField = iprot.readFieldBegin();
        if (schemeField.type == org.apache.thrift.protocol.TType.STOP) { 
          break;
        }
        switch (schemeField.id) {
          case 1: // TEXT
            if (schemeField.type == org.apache.thrift.protocol.TType.STRING) {
              struct.text = iprot.readString();
              struct.setTextIsSet(true);
            } else { 
              org.apache.thrift.protocol.TProtocolUtil.skip(iprot, schemeField.type);
            }
            break;
          case 2: // HIGHLIGHTS
            if (schemeField.type == org.apache.thrift.protocol.TType.LIST) {
              {
                org.apache.thrift.protocol.TList _list24 = iprot.readListBegin();
                struct.highlights = new ArrayList<TextRegion>(_list24.size);
                for (int _i25 = 0; _i25 < _list24.size; ++_i25)
                {
                  TextRegion _elem26; // optional
                  _elem26 = new TextRegion();
                  _elem26.read(iprot);
                  struct.highlights.add(_elem26);
                }
                iprot.readListEnd();
              }
              struct.setHighlightsIsSet(true);
            } else { 
              org.apache.thrift.protocol.TProtocolUtil.skip(iprot, schemeField.type);
            }
            break;
          default:
            org.apache.thrift.protocol.TProtocolUtil.skip(iprot, schemeField.type);
        }
        iprot.readFieldEnd();
      }
      iprot.readStructEnd();

      // check for required fields of primitive type, which can't be checked in the validate method
      struct.validate();
    }

    public void write(org.apache.thrift.protocol.TProtocol oprot, ResultSummary struct) throws org.apache.thrift.TException {
      struct.validate();

      oprot.writeStructBegin(STRUCT_DESC);
      if (struct.text != null) {
        oprot.writeFieldBegin(TEXT_FIELD_DESC);
        oprot.writeString(struct.text);
        oprot.writeFieldEnd();
      }
      if (struct.highlights != null) {
        oprot.writeFieldBegin(HIGHLIGHTS_FIELD_DESC);
        {
          oprot.writeListBegin(new org.apache.thrift.protocol.TList(org.apache.thrift.protocol.TType.STRUCT, struct.highlights.size()));
          for (TextRegion _iter27 : struct.highlights)
          {
            _iter27.write(oprot);
          }
          oprot.writeListEnd();
        }
        oprot.writeFieldEnd();
      }
      oprot.writeFieldStop();
      oprot.writeStructEnd();
    }

  }

  private static class ResultSummaryTupleSchemeFactory implements SchemeFactory {
    public ResultSummaryTupleScheme getScheme() {
      return new ResultSummaryTupleScheme();
    }
  }

  private static class ResultSummaryTupleScheme extends TupleScheme<ResultSummary> {

    @Override
    public void write(org.apache.thrift.protocol.TProtocol prot, ResultSummary struct) throws org.apache.thrift.TException {
      TTupleProtocol oprot = (TTupleProtocol) prot;
      BitSet optionals = new BitSet();
      if (struct.isSetText()) {
        optionals.set(0);
      }
      if (struct.isSetHighlights()) {
        optionals.set(1);
      }
      oprot.writeBitSet(optionals, 2);
      if (struct.isSetText()) {
        oprot.writeString(struct.text);
      }
      if (struct.isSetHighlights()) {
        {
          oprot.writeI32(struct.highlights.size());
          for (TextRegion _iter28 : struct.highlights)
          {
            _iter28.write(oprot);
          }
        }
      }
    }

    @Override
    public void read(org.apache.thrift.protocol.TProtocol prot, ResultSummary struct) throws org.apache.thrift.TException {
      TTupleProtocol iprot = (TTupleProtocol) prot;
      BitSet incoming = iprot.readBitSet(2);
      if (incoming.get(0)) {
        struct.text = iprot.readString();
        struct.setTextIsSet(true);
      }
      if (incoming.get(1)) {
        {
          org.apache.thrift.protocol.TList _list29 = new org.apache.thrift.protocol.TList(org.apache.thrift.protocol.TType.STRUCT, iprot.readI32());
          struct.highlights = new ArrayList<TextRegion>(_list29.size);
          for (int _i30 = 0; _i30 < _list29.size; ++_i30)
          {
            TextRegion _elem31; // optional
            _elem31 = new TextRegion();
            _elem31.read(iprot);
            struct.highlights.add(_elem31);
          }
        }
        struct.setHighlightsIsSet(true);
      }
    }
  }

}

