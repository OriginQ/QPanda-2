// The MIT License (MIT)
//
// Copyright (c) 2013-2014 mashiro
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#ifndef RABBIT_HPP_INCLUDED
#define RABBIT_HPP_INCLUDED

#ifdef __clang__
#pragma clang diagnostic ignored "-Wtautological-constant-out-of-range-compare"
#endif

#ifndef RABBIT_NAMESPACE
#define RABBIT_NAMESPACE rabbit
#endif

#include <string>
#include <stdexcept>
#include <iterator>
#include <algorithm>
#include <sstream>
#include <iosfwd>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/error/en.h>

namespace RABBIT_NAMESPACE {

#define RABBIT_TAG_DEF(name, id) \
  struct name \
  { \
    static const rapidjson::Type native_value = id; \
    static const int value = id; \
  }; \
/**/
RABBIT_TAG_DEF(null_tag, rapidjson::kNullType)
RABBIT_TAG_DEF(false_tag, rapidjson::kFalseType)
RABBIT_TAG_DEF(true_tag, rapidjson::kTrueType)
RABBIT_TAG_DEF(object_tag, rapidjson::kObjectType)
RABBIT_TAG_DEF(array_tag, rapidjson::kArrayType)
RABBIT_TAG_DEF(string_tag, rapidjson::kStringType)
RABBIT_TAG_DEF(number_tag, rapidjson::kNumberType)
#undef RABBIT_TAG_DEF

class type_mismatch : public std::runtime_error
{
public:
  type_mismatch(const std::string& msg)
    : std::runtime_error(msg)
  {}
};

typedef rapidjson::ParseErrorCode parse_error_code;

class parse_error : public std::runtime_error
{
private:
  parse_error_code code_;

public:
  parse_error(parse_error_code code)
    : std::runtime_error(rapidjson::GetParseError_En(code))
    , code_(code)
  {}

  parse_error_code code() const { return code_; }
};

// fwd
template <typename Traits> class basic_value_ref;
template <typename Traits, typename DefaultTag> class basic_value;
template <typename Traits> class basic_object;
template <typename Traits> class basic_array;

namespace details {

  template <bool C, typename T = void>
  struct enable_if_c
  {
    typedef T type;
  };

  template <typename T>
  struct enable_if_c<false, T>
  {};

  template <typename Cond, typename T = void>
  struct enable_if : enable_if_c<Cond::value, T>
  {};

  template <typename Cond, typename T = void>
  struct disable_if : enable_if_c<!Cond::value, T>
  {};


  template <bool C>
  struct bool_
  {
    static const bool value = C;
  };

  typedef bool_<true> true_;
  typedef bool_<false> false_;


  template <typename T> struct remove_reference { typedef T type; };
  template <typename T> struct remove_reference<T&> { typedef T type; };

  template <typename T> struct remove_const { typedef T type; };
  template <typename T> struct remove_const<const T> { typedef T type; };


  template <typename Char, typename Traits = std::char_traits<Char> >
  class basic_string_ref
  {
  public:
    typedef Char value_type;
    typedef std::size_t size_type;

  private:
    const value_type* data_;
    size_type length_;

  public:
    basic_string_ref()
      : data_(0)
      , length_(0)
    {}

    basic_string_ref(const basic_string_ref& other)
      : data_(other.data_)
      , length_(other.length_)
    {}

    basic_string_ref(const value_type* str)
      : data_(str)
      , length_(Traits::length(str))
    {}

    basic_string_ref(const value_type* str, size_type length)
      : data_(str)
      , length_(length)
    {}

    template <typename Allocator>
    basic_string_ref(const std::basic_string<value_type, Allocator>& other)
      : data_(other.data())
      , length_(other.length())
    {}

    size_type size() const     { return length_; }
    size_type length() const   { return length_; }
    size_type max_size() const { return length_; }
    bool empty() const         { return length_ == 0; }

    const value_type* data() const { return data_; }
  };


  // type traits
  template <typename T> struct is_tag : false_ {};
  template <> struct is_tag<null_tag> : true_ {};
  template <> struct is_tag<false_tag> : true_ {};
  template <> struct is_tag<true_tag> : true_ {};
  template <> struct is_tag<object_tag> : true_ {};
  template <> struct is_tag<array_tag> : true_ {};
  template <> struct is_tag<string_tag> : true_ {};
  template <> struct is_tag<number_tag> : true_ {};

  template <typename T> struct is_null : false_ {};
  template <> struct is_null<null_tag> : true_ {};

  template <typename T> struct is_false : false_ {};
  template <> struct is_false<false_tag> : true_ {};

  template <typename T> struct is_true : false_ {};
  template <> struct is_true<true_tag> : true_ {};

  template <typename T> struct is_object : false_ {};
  template <> struct is_object<object_tag> : true_ {};
  template <typename Traits> struct is_object< basic_object<Traits> > : true_ {};

  template <typename T> struct is_array : false_ {};
  template <> struct is_array<array_tag> : true_ {};
  template <typename Traits> struct is_array< basic_array<Traits> > : true_ {};

  template <typename T> struct is_string : false_ {};
  template <> struct is_string<string_tag> : true_ {};
  template <typename Char> struct is_string< std::basic_string<Char> > : true_ {};
  template <typename Char, typename Traits> struct is_string< basic_string_ref<Char, Traits> > : true_ {};

  template <typename T> struct is_cstr_ptr : false_ {};
  template <> struct is_cstr_ptr< char * > : true_ {};
  template <> struct is_cstr_ptr< const char * > : true_ {};
  template <> struct is_cstr_ptr< wchar_t * > : true_ {};
  template <> struct is_cstr_ptr< const wchar_t * > : true_ {};

  template <typename T> struct is_number : false_ {};
  template <> struct is_number<number_tag> : true_ {};

  template <typename T> struct is_bool : false_ {};
  template <> struct is_bool<bool> : true_ {};

  template <typename T> struct is_int : false_ {};
  template <> struct is_int<int> : true_ {};

  template <typename T> struct is_uint : false_ {};
  template <> struct is_uint<unsigned> : true_ {};

  template <typename T> struct is_int64 : false_ {};
  template <> struct is_int64<int64_t> : true_ {};

  template <typename T> struct is_uint64 : false_ {};
  template <> struct is_uint64<uint64_t> : true_ {};

  template <typename T> struct is_double : false_ {};
  template <> struct is_double<double> : true_ {};

  template <typename T> struct is_value_ref : false_ {};
  template <typename Traits> struct is_value_ref< basic_value_ref<Traits> > : true_ {};
  template <typename Traits, typename DefaultTag> struct is_value_ref< basic_value<Traits, DefaultTag> > : true_ {};
  template <typename Traits> struct is_value_ref< basic_object<Traits> > : true_ {};
  template <typename Traits> struct is_value_ref< basic_array<Traits> > : true_ {};

  // type name
  template <typename T> const char* type_name(typename enable_if< is_null<T> >::type* = 0)        { return "null"; }
  template <typename T> const char* type_name(typename enable_if< is_false<T> >::type* = 0)       { return "false"; }
  template <typename T> const char* type_name(typename enable_if< is_true<T> >::type* = 0)        { return "true"; }
  template <typename T> const char* type_name(typename enable_if< is_object<T> >::type* = 0)      { return "object"; }
  template <typename T> const char* type_name(typename enable_if< is_array<T> >::type* = 0)       { return "array"; }
  template <typename T> const char* type_name(typename enable_if< is_string<T> >::type* = 0)      { return "string"; }
  template <typename T> const char* type_name(typename enable_if< is_number<T> >::type* = 0)      { return "number"; }
  template <typename T> const char* type_name(typename enable_if< is_bool<T> >::type* = 0)        { return "bool"; }
  template <typename T> const char* type_name(typename enable_if< is_int<T> >::type* = 0)         { return "int"; }
  template <typename T> const char* type_name(typename enable_if< is_uint<T> >::type* = 0)        { return "uint"; }
  template <typename T> const char* type_name(typename enable_if< is_int64<T> >::type* = 0)       { return "int64"; }
  template <typename T> const char* type_name(typename enable_if< is_uint64<T> >::type* = 0)      { return "uint64"; }
  template <typename T> const char* type_name(typename enable_if< is_double<T> >::type* = 0)      { return "double"; }
  template <typename T> const char* type_name(typename enable_if< is_value_ref<T> >::type* = 0)   { return "value_ref"; }


  template <typename PseudoReference>
  struct operator_arrow_proxy
  {
    mutable typename remove_const<PseudoReference>::type value_;
    operator_arrow_proxy(const PseudoReference& value) : value_(value) {}
    PseudoReference* operator->() const { return &value_; }
  };

  template <typename T>
  struct operator_arrow_proxy<T&>
  {
    T& value_;
    operator_arrow_proxy(T& value) : value_(value) {}
    T* operator->() const { return &value_; }
  };


  template <typename Function, typename Iterator>
  class transform_iterator
  {
    typedef std::iterator_traits<Iterator> traits_type;

  public:
    typedef transform_iterator this_type;
    typedef Function function_type;

    typedef Iterator iterator_type;
    typedef typename traits_type::iterator_category iterator_category;
    typedef typename traits_type::difference_type difference_type;

    typedef typename Function::result_type result_type;
    typedef typename remove_reference<result_type>::type value_type;
    typedef operator_arrow_proxy<result_type> operator_arrow_proxy_type;
    typedef operator_arrow_proxy_type pointer;
    typedef result_type reference;

  private:
    iterator_type it_;
    function_type func_;

  public:
    transform_iterator()
      : it_()
      , func_()
    {}

    explicit transform_iterator(const iterator_type& it)
      : it_(it)
      , func_()
    {}

    transform_iterator(const iterator_type& it, const function_type& func)
      : it_(it)
      , func_(func)
    {}

    template <typename OtherFunction, typename OtherIterator>
    transform_iterator(const transform_iterator<OtherFunction, OtherIterator>& other)
      : it_(other.base())
      , func_(other.functor())
    {}

    iterator_type& base() { return it_; }
    const iterator_type& base() const { return it_; }

    function_type& functor() { return func_; }
    const function_type& functor() const { return func_; }

    result_type dereference() const { return func_(*it_); }

    result_type operator*() const { return dereference(); }
    operator_arrow_proxy_type operator->() const { return operator_arrow_proxy_type(dereference()); }

    this_type& operator++() { ++it_; return *this; }
    this_type operator++(int) { return this_type(it_++, func_); }
    this_type& operator--() { --it_; return *this; }
    this_type operator--(int) { return this_type(it_--, func_); }

    this_type operator+(difference_type n) const { return this_type(it_ + n, func_); }
    this_type& operator+=(difference_type n) { it_ += n; return *this; }
    this_type operator-(difference_type n) const { return this_type(it_ - n, func_); }
    this_type& operator-=(difference_type n) { it_ -= n; return *this; }

    result_type operator[](difference_type n) const { return func_(it_[n]); }

    template <typename OtherFunction, typename OtherIterator>
    bool operator==(const transform_iterator<OtherFunction, OtherIterator>& other) const { return base() == other.base(); }

    template <typename OtherFunction, typename OtherIterator>
    bool operator!=(const transform_iterator<OtherFunction, OtherIterator>& other) const { return base() != other.base(); }

    template <typename OtherFunction, typename OtherIterator>
    bool operator<(const transform_iterator<OtherFunction, OtherIterator>& other) const { return base() < other.base(); }

    template <typename OtherFunction, typename OtherIterator>
    bool operator>(const transform_iterator<OtherFunction, OtherIterator>& other) const { return base() > other.base(); }

    template <typename OtherFunction, typename OtherIterator>
    bool operator<=(const transform_iterator<OtherFunction, OtherIterator>& other) const { return base() <= other.base(); }

    template <typename OtherFunction, typename OtherIterator>
    bool operator>=(const transform_iterator<OtherFunction, OtherIterator>& other) const { return base() >= other.base(); }
  };

  template <typename Function, typename Iterator>
  transform_iterator<Function, Iterator> make_transform_iterator(Iterator it, Function func = Function())
  {
    return transform_iterator<Function, Iterator>(it, func);
  }


  template <typename Member, typename ValueRef>
  class member_wrapper
  {
  public:
    typedef Member wrapped_type;
    typedef ValueRef value_ref_type;
    typedef typename ValueRef::string_type string_type;
    typedef typename ValueRef::allocator_type allocator_type;

    class proxy
    {
      wrapped_type& member_;
      allocator_type* alloc_;

    public:
      proxy(wrapped_type& member, allocator_type* alloc)
        : member_(member)
        , alloc_(alloc)
      {}

      string_type name() const { return value_ref_type(&(member_.name), alloc_).as_string(); }
      value_ref_type value() const { return value_ref_type(&(member_.value), alloc_); }
    };

  private:
    allocator_type* alloc_;

  public:
    member_wrapper(allocator_type* alloc)
      : alloc_(alloc)
    {}

    template <typename OtherMember, typename OtherValueRef>
    member_wrapper(const member_wrapper<OtherMember, OtherValueRef>& other)
      : alloc_(other.get_allocator_pointer())
    {}

    typedef proxy result_type;
    result_type operator()(wrapped_type& member) const
    {
      return result_type(member, alloc_);
    }

    allocator_type* get_allocator_pointer() const { return alloc_; }
  };

  template <typename Value, typename ValueRef>
  class value_wrapper
  {
  public:
    typedef Value wrapped_type;
    typedef ValueRef value_ref_type;
    typedef typename ValueRef::string_type string_type;
    typedef typename ValueRef::allocator_type allocator_type;

  private:
    allocator_type* alloc_;

  public:
    value_wrapper(allocator_type* alloc)
      : alloc_(alloc)
    {}

    template <typename OtherValue, typename OtherValueRef>
    value_wrapper(const value_wrapper<OtherValue, OtherValueRef>& other)
      : alloc_(other.get_allocator_pointer())
    {}

    typedef value_ref_type result_type;
    result_type operator()(wrapped_type& value) const
    {
      return result_type(&value, alloc_);
    }

    allocator_type* get_allocator_pointer() const { return alloc_; }
  };

  template <typename Encoding> struct value_ref_traits;
  template <typename Encoding> struct const_value_ref_traits;

  template <typename Encoding>
  struct value_ref_traits
  {
    typedef Encoding                                            encoding_type;
    typedef rapidjson::Type                                     native_type;
    typedef rapidjson::GenericDocument<Encoding>                native_document_type;
    typedef rapidjson::GenericValue<Encoding>                   native_value_type;
    typedef typename native_document_type::AllocatorType        native_allocator_type;
    typedef const_value_ref_traits<Encoding>                    const_traits;

    template <typename ValueRef, typename Tag>
    static void set(ValueRef& ref, Tag tag = Tag())
    {
      ref.set(tag);
    }
  };

  template <typename Encoding>
  struct const_value_ref_traits
  {
    typedef Encoding                                            encoding_type;
    typedef const rapidjson::Type                               native_type;
    typedef const rapidjson::GenericDocument<Encoding>          native_document_type;
    typedef const rapidjson::GenericValue<Encoding>             native_value_type;
    typedef const typename native_document_type::AllocatorType  native_allocator_type;
    typedef const_value_ref_traits<Encoding>                    const_traits;

    template <typename ValueRef, typename Tag>
    static void set(const ValueRef& ref, Tag tag = Tag())
    {}
  };


  template <typename T>
  class scoped_ptr
  {
  private:
    T* p_;

  private:
    scoped_ptr(const scoped_ptr& other);
    scoped_ptr& operator=(const scoped_ptr& other);

  public:
    explicit scoped_ptr(T* p = 0)
      : p_(p)
    {}

    ~scoped_ptr()
    {
      delete p_;
    }

    T* operator->() { return p_; }
    const T* operator->() const { return p_; }

    T& operator*() { return *p_; }
    const T& operator*() const { return *p_; }

    T* get() { return p_; }
    const T* get() const { return p_; }

    void swap(scoped_ptr& other) throw()
    {
      std::swap(p_, other.p_);
    }
  };

} // details

template <typename Traits>
class basic_value_ref
{
public:
  typedef Traits                                        traits;
  typedef typename Traits::const_traits                 const_traits;

  typedef typename traits::encoding_type                encoding_type;
  typedef typename traits::native_type                  native_type;
  typedef typename traits::native_document_type         native_document_type;
  typedef typename traits::native_value_type            native_value_type;
  typedef typename traits::native_allocator_type        native_allocator_type;

  typedef basic_value_ref<traits>                       value_ref_type;
  typedef const basic_value_ref<const_traits>           const_value_ref_type;
  typedef typename encoding_type::Ch                    char_type;
  typedef std::basic_string<char_type>                  string_type;
  typedef details::basic_string_ref<char_type>          string_ref_type;
  typedef native_allocator_type                         allocator_type;

private:
  typedef details::member_wrapper<      typename native_value_type::Member,       value_ref_type> member_wrapper_type;
  typedef details::member_wrapper<const typename native_value_type::Member, const_value_ref_type> const_member_wrapper_type;
  typedef details::value_wrapper<      native_value_type,       value_ref_type> value_wrapper_type;
  typedef details::value_wrapper<const native_value_type, const_value_ref_type> const_value_wrapper_type;

public:
  typedef details::transform_iterator<      member_wrapper_type, typename native_value_type::MemberIterator> member_iterator;
  typedef details::transform_iterator<const_member_wrapper_type, typename native_value_type::ConstMemberIterator> const_member_iterator;
  typedef details::transform_iterator<      value_wrapper_type, typename native_value_type::ValueIterator> value_iterator;
  typedef details::transform_iterator<const_value_wrapper_type, typename native_value_type::ConstValueIterator> const_value_iterator;

private:
  native_value_type* value_;
  allocator_type* alloc_;

public:
  basic_value_ref(native_value_type* value = 0, allocator_type* alloc = 0)
    : value_(value)
    , alloc_(alloc)
  {}

  template <typename OtherTraits>
  basic_value_ref(const basic_value_ref<OtherTraits>& other)
    : value_(other.get_native_value_pointer())
    , alloc_(other.get_allocator_pointer())
  {}

  native_value_type* get_native_value_pointer() const { return value_; }
  allocator_type* get_allocator_pointer() const { return alloc_; }
  allocator_type& get_allocator() const { return *alloc_; }

  void set(null_tag)                  { value_->SetNull(); }
  void set(object_tag)                { value_->SetObject(); }
  void set(array_tag)                 { value_->SetArray(); }
  void set(bool value)                { value_->SetBool(value); }
  void set(int value)                 { value_->SetInt(value); }
  void set(unsigned value)            { value_->SetUint(value); }
  void set(int64_t value)             { value_->SetInt64(value); }
  void set(uint64_t value)            { value_->SetUint64(value); }
  void set(double value)              { value_->SetDouble(value); }
  void set(const char_type* value)    { value_->SetString(value, *alloc_); }
  void set(const string_type& value)  { value_->SetString(value.data(), value.length(), *alloc_); }

  template <typename T>
  void set(const T& value, typename details::enable_if< details::is_value_ref<T> >::type* = 0)
  {
    if      (value.is_null())   set(null_tag());
    else if (value.is_bool())   set(value.as_bool());
    else if (value.is_int())    set(value.as_int());
    else if (value.is_uint())   set(value.as_uint());
    else if (value.is_int64())  set(value.as_int64());
    else if (value.is_uint64()) set(value.as_uint64());
    else if (value.is_double()) set(value.as_double());
    else if (value.is_string()) set(value.as_string());
    else if (value.is_array()) throw std::runtime_error("can not assign array directly. please use insert");
    else if (value.is_object()) throw std::runtime_error("can not assign object directly. please use insert");
  }

  template <typename OtherTraits> 
  void deep_copy(const basic_value_ref<OtherTraits>& other)
  {
    value_->CopyFrom(*other.get_native_value_pointer(), *alloc_);
  }


  template <typename T>
  value_ref_type& operator=(const T& value)
  {
    set(value);
    return *this;
  }


  template <typename OtherTraits>
  bool operator==(const basic_value_ref<OtherTraits>& other) const
  {
    if (is_null() && other.is_null()) return true;
    if (is_bool() && other.is_bool() && as_bool() == other.as_bool()) return true;
    if (is_int() && other.is_int() && as_int() == other.as_int()) return true;
    if (is_uint() && other.is_uint() && as_uint() == other.as_uint()) return true;
    if (is_int64() && other.is_int64() && as_int64() == other.as_int64()) return true;
    if (is_uint64() && other.is_uint64() && as_uint64() == other.as_uint64()) return true;
    if (is_double() && other.is_double() && as_double() == other.as_double()) return true;
    if (is_string() && other.is_string() && as_string() == other.as_string()) return true;
    return false;
  }

  template <typename OtherTraits>
  bool operator!=(const basic_value_ref<OtherTraits>& other) const
  {
    return !(*this == other);
  }


  int which() const
  {
    return static_cast<int>(value_->GetType());
  }

#define RABBIT_IS_DEF(name, base_name) \
  template <typename T> \
  bool is(typename details::enable_if< details::is_##name<T> >::type* = 0) const \
  { \
    return value_->Is##base_name(); \
  } \
  bool is_##name() const \
  { \
    return value_->Is##base_name(); \
  } \
/**/
  RABBIT_IS_DEF(null, Null)
  RABBIT_IS_DEF(false, False)
  RABBIT_IS_DEF(true, True)
  RABBIT_IS_DEF(object, Object)
  RABBIT_IS_DEF(array, Array)
  RABBIT_IS_DEF(number, Number)
  RABBIT_IS_DEF(bool, Bool)
  RABBIT_IS_DEF(int, Int)
  RABBIT_IS_DEF(uint, Uint)
  RABBIT_IS_DEF(int64, Int64)
  RABBIT_IS_DEF(uint64, Uint64)
  RABBIT_IS_DEF(double, Double)
  RABBIT_IS_DEF(string, String)
#undef RABBIT_IS_DEF

#define RABBIT_AS_DEF(result_type, name, base_name) \
  template <typename T> \
  T as(typename details::enable_if< details::is_##name<T> >::type* = 0) const \
  { \
    type_check<T>(); \
    return value_->Get##base_name(); \
  } \
  result_type as_##name() const \
  { \
    type_check<result_type>(); \
    return value_->Get##base_name(); \
  } \
/**/
  RABBIT_AS_DEF(bool, bool, Bool)
  RABBIT_AS_DEF(int, int, Int)
  RABBIT_AS_DEF(unsigned, uint, Uint)
  RABBIT_AS_DEF(int64_t, int64, Int64)
  RABBIT_AS_DEF(uint64_t, uint64, Uint64)
  RABBIT_AS_DEF(double, double, Double)
  RABBIT_AS_DEF(string_type, string, String)
#undef RABBIT_AS_DEF

private:
  struct as_t
  {
    const value_ref_type& ref_;
    as_t(const value_ref_type& ref) : ref_(ref) {}

    template <typename Result>
    operator Result() const { return ref_.as<Result>(); }
  };

public:
  as_t as() const { return as_t(*this); }

  bool has(const string_ref_type& name) const
  {
    type_check<object_tag>();
    return value_->HasMember(name.data());
  }

  template <typename T>
  void insert(const string_ref_type& name, const T& value, const bool copy_name_string = true, typename details::disable_if< details::is_value_ref<T> >::type* = 0, typename details::enable_if< details::is_string<T> >::type * = 0)
  {
    type_check<object_tag>();
    native_value_type v(value.data(), value.length(), *alloc_);
    if(copy_name_string){
      native_value_type copied_name(name.data(), name.length(), *alloc_);
      value_->AddMember(copied_name, v, *alloc_);
    }else{
      value_->AddMember(rapidjson::StringRef(name.data(), name.length()), v, *alloc_);
    }
  }

  template <typename T>
  void insert(const string_ref_type& name, const T& value, const bool copy_name_string = true, typename details::disable_if< details::is_value_ref<T> >::type* = 0, typename details::enable_if< details::is_cstr_ptr<T> >::type * = 0)
  {
    type_check<object_tag>();
    native_value_type v(value, *alloc_);
    if(copy_name_string){
      native_value_type copied_name(name.data(), name.length(), *alloc_);
      value_->AddMember(copied_name, v, *alloc_);
    }else{
      value_->AddMember(rapidjson::StringRef(name.data(), name.length()), v, *alloc_);
    }
  }


  template <typename T>
  void insert(const string_ref_type& name, const T& value, const bool copy_name_string = true, typename details::disable_if< details::is_value_ref<T> >::type* = 0, typename details::disable_if< details::is_string<T> >::type * = 0, typename details::disable_if< details::is_cstr_ptr<T> >::type * = 0)
  {
    type_check<object_tag>();
    native_value_type v(value);
    if(copy_name_string){
      native_value_type copied_name(name.data(), name.length(), *alloc_);
      value_->AddMember(copied_name, v, *alloc_);
    }else{
      value_->AddMember(rapidjson::StringRef(name.data(), name.length()), v, *alloc_);
    }
  }

  template <typename T>
  void insert(const string_ref_type& name, const T& value, const bool copy_name_string = true, typename details::enable_if< details::is_value_ref<T> >::type* = 0)
  {
    type_check<object_tag>();
    if(copy_name_string){
      native_value_type copied_name(name.data(), name.length(), *alloc_);
      value_->AddMember(copied_name, *value.get_native_value_pointer(), *alloc_);
    }else{
      value_->AddMember(rapidjson::StringRef(name.data(), name.length()), *value.get_native_value_pointer(), *alloc_);
    }

  }

  bool erase(const string_ref_type& name)
  {
    type_check<object_tag>();
    return value_->RemoveMember(name.data());
  }

  void reserve(const size_t reserve_size){
    type_check<array_tag>();
    value_->Reserve(reserve_size, *alloc_);
  }

  const_member_iterator erase(const const_member_iterator& itr){
    type_check<object_tag>();
    return details::make_transform_iterator(value_->EraseMember(itr.base()), const_member_wrapper_type(alloc_));
  }

  member_iterator erase(const member_iterator& itr){
    type_check<object_tag>();
    return details::make_transform_iterator(value_->EraseMember(itr.base()), member_wrapper_type(alloc_));
  }

  const_value_iterator erase(const const_value_iterator& itr){
    type_check<array_tag>();
    return details::make_transform_iterator(value_->Erase(itr.base()), const_value_wrapper_type(alloc_));
  }

  const_value_iterator erase(const const_value_iterator& beginItr, const const_value_iterator& endItr){
    type_check<array_tag>();
    return details::make_transform_iterator(value_->Erase(beginItr.base(), endItr.base()), const_value_wrapper_type(alloc_));
  }

  value_iterator erase(const value_iterator& itr){
    type_check<array_tag>();
    return details::make_transform_iterator(value_->Erase(itr.base()), value_wrapper_type(alloc_));
  }

  value_iterator erase(const value_iterator& beginItr, const value_iterator& endItr){
    type_check<array_tag>();
    return details::make_transform_iterator(value_->Erase(beginItr.base(), endItr.base()), value_wrapper_type(alloc_));
  }

  value_ref_type at(const string_ref_type& name)
  {
    type_check<object_tag>();

    if (!has(name))
    {
      native_value_type null;
      native_value_type copied_name(name.data(), name.length(), *alloc_);
      value_->AddMember(copied_name, null, *alloc_);
    }

    return value_ref_type(&((*value_)[name.data()]), alloc_);
  }

  const_value_ref_type at(const string_ref_type& name) const
  {
    type_check<object_tag>();
    if (!has(name))
      throw std::out_of_range("'" + string_type(name.data(), name.size()) + "' not found");
    return const_value_ref_type(&((*value_)[name.data()]), alloc_);
  }

  value_ref_type operator[](const string_ref_type& name) { return at(name); }
  const_value_ref_type operator[](const string_ref_type& name) const { return at(name); }

  member_iterator member_begin()
  {
    type_check<object_tag>();
    return details::make_transform_iterator(value_->MemberBegin(), member_wrapper_type(alloc_));
  }

  member_iterator member_end()
  {
    type_check<object_tag>();
    return details::make_transform_iterator(value_->MemberEnd(), member_wrapper_type(alloc_));
  }

  const_member_iterator member_begin() const
  {
    type_check<object_tag>();
    return details::make_transform_iterator(value_->MemberBegin(), const_member_wrapper_type(alloc_));
  }

  const_member_iterator member_end() const
  {
    type_check<object_tag>();
    return details::make_transform_iterator(value_->MemberEnd(), const_member_wrapper_type(alloc_));
  }

  const_member_iterator member_cbegin() const { return member_begin(); }
  const_member_iterator member_cend() const { return member_end(); }

  std::size_t size() const
  {
    if (is_object()) {
      return value_->MemberCount();
    } else if (is_array()) {
      return value_->Size();
    }
    throw type_mismatch("cannot take size of non-object/array");
  }

  std::size_t capacity() const
  {
    type_check<array_tag>();
    return value_->Capacity();
  }

  bool empty() const
  {
    type_check<array_tag>();
    return value_->Empty();
  }

  value_ref_type at(std::size_t index)
  {
    type_check<array_tag>();
    range_check(index);
    return value_ref_type(&((*value_)[index]), alloc_);
  }

  const_value_ref_type at(std::size_t index) const
  {
    type_check<array_tag>();
    range_check(index);
    return const_value_ref_type(&((*value_)[index]), alloc_);
  }

  value_ref_type operator[](std::size_t index) { return at(index); }
  const_value_ref_type operator[](std::size_t index) const { return at(index); }

  template <typename T>
  void push_back(const T& value, typename details::disable_if< details::is_value_ref<T> >::type* = 0, typename details::enable_if< details::is_string<T> >::type * = 0)
  {
    type_check<array_tag>();
    native_value_type v(value.data(), value.length(), *alloc_);
    value_->PushBack(v, *alloc_);
  }

  template <typename T>
  void push_back(const T& value, typename details::disable_if< details::is_value_ref<T> >::type* = 0, typename details::enable_if< details::is_cstr_ptr<T> >::type * = 0)
  {
    type_check<array_tag>();
    native_value_type v(value, *alloc_);
    value_->PushBack(v, *alloc_);
  }

  template <typename T>
  void push_back(const T& value, typename details::disable_if< details::is_value_ref<T> >::type* = 0, typename details::disable_if< details::is_string<T> >::type * = 0, typename details::disable_if< details::is_cstr_ptr<T> >::type * = 0)
  {
    type_check<array_tag>();
    native_value_type v(value);
    value_->PushBack(v, *alloc_);
  }

  template <typename T>
  void push_back(const T& value, typename details::enable_if< details::is_value_ref<T> >::type* = 0)
  {
      type_check<array_tag>();
#if NOT_DEEP_COPY
      value_->PushBack(*value.get_native_value_pointer(), *alloc_);
#else
      native_value_type v;
      v.CopyFrom(*value.get_native_value_pointer(), *alloc_);
      value_->PushBack(v, *alloc_);
#endif
  }

  void pop_back()
  {
    type_check<array_tag>();
    value_->PopBack();
  }

  value_iterator value_begin()
  {
    type_check<array_tag>();
    return details::make_transform_iterator(value_->Begin(), value_wrapper_type(alloc_));
  }

  value_iterator value_end()
  {
    type_check<array_tag>();
    return details::make_transform_iterator(value_->End(), value_wrapper_type(alloc_));
  }

  const_value_iterator value_begin() const
  {
    type_check<array_tag>();
    return details::make_transform_iterator(value_->Begin(), const_value_wrapper_type(alloc_));
  }

  const_value_iterator value_end() const
  {
    type_check<array_tag>();
    return details::make_transform_iterator(value_->End(), const_value_wrapper_type(alloc_));
  }

  const_value_iterator value_cbegin() const { return value_begin(); }
  const_value_iterator value_cend() const { return value_end(); }


  void swap(value_ref_type& other) throw()
  {
    std::swap(value_, other.value_);
    std::swap(alloc_, other.alloc_);
  }

  string_type str() const
  {
    switch (which())
    {
    case null_tag::value:
      return "null";

    case false_tag::value:
      return "false";

    case true_tag::value:
      return "true";

    case string_tag::value:
      return as_string();

    case number_tag::value:
      {
        std::basic_stringstream<char_type> ss;
        if      (is_int())    ss << as_int();
        else if (is_uint())   ss << as_uint();
        else if (is_int64())  ss << as_int64();
        else if (is_uint64()) ss << as_uint64();
        else if (is_double()) ss << as_double();
        return ss.str();
      }

    default:
      {
        typedef rapidjson::GenericStringBuffer<encoding_type> buffer_t;
        typedef rapidjson::Writer<buffer_t, encoding_type> writer_t;
        buffer_t buffer;
        writer_t writer(buffer);
        value_->Accept(writer);
        return buffer.GetString();
      }
    }
  }

private:
  template <typename T>
  void type_check() const
  {
    if (!is<T>())
    {
      std::stringstream ss;
      ss << "value is not ";
      ss << details::type_name<T>();
      ss << " (which is " << which() << ")";
      throw type_mismatch(ss.str());
    }
  }

  void range_check(std::size_t index) const
  {
    if (index >= size())
    {
      std::stringstream ss;
      ss << "index (which is " << index << ") >= size() (which is " << size() << ")";
      throw std::out_of_range(ss.str());
    }
  }
};


template <typename Traits>
struct basic_value_base
{
  typedef basic_value_ref<Traits>                   base_type;
  typedef typename base_type::native_value_type     native_value_type;
  typedef typename base_type::allocator_type        allocator_type;

  details::scoped_ptr<native_value_type> value_impl_;
  details::scoped_ptr<allocator_type> alloc_impl_;

  explicit basic_value_base(native_value_type* value = 0, allocator_type* alloc = 0)
    : value_impl_(value)
    , alloc_impl_(alloc)
  {}
};

template <typename Traits, typename DefaultTag = null_tag>
class basic_value 
  : private basic_value_base<Traits>
  , public basic_value_ref<Traits>
{
public:
  typedef Traits                                        traits;
  typedef typename Traits::const_traits                 const_traits;

  typedef basic_value_base<traits>                      member_type;
  typedef basic_value_ref<traits>                       base_type;

  typedef typename base_type::encoding_type             encoding_type;
  typedef typename base_type::native_type               native_type;
  typedef typename base_type::native_document_type      native_document_type;
  typedef typename base_type::native_value_type         native_value_type;
  typedef typename base_type::native_allocator_type     native_allocator_type;

  typedef typename base_type::value_ref_type            value_ref_type;
  typedef typename base_type::const_value_ref_type      const_value_ref_type;
  typedef typename base_type::char_type                 char_type;
  typedef typename base_type::string_type               string_type;
  typedef typename base_type::string_ref_type           string_ref_type;
  typedef typename base_type::allocator_type            allocator_type;

  typedef typename base_type::member_iterator           member_iterator;
  typedef typename base_type::const_member_iterator     const_member_iterator;
  typedef typename base_type::value_iterator            value_iterator;
  typedef typename base_type::const_value_iterator      const_value_iterator;

private:
  typedef DefaultTag default_tag;

public:
  basic_value()
    : member_type(new native_value_type(DefaultTag::native_value), new allocator_type())
    , base_type(member_type::value_impl_.get(), member_type::alloc_impl_.get())
  {}

  basic_value(allocator_type& alloc)
    : member_type(new native_value_type(DefaultTag::native_value))
    , base_type(member_type::value_impl_.get(), &alloc)
  {}


  /*
   *        Tag based constructors  
   */
  template <typename Tag>
  basic_value(Tag tag, typename details::enable_if< details::is_tag<Tag> >::type* = 0)
    : member_type(new native_value_type(Tag::native_value), new allocator_type())
    , base_type(member_type::value_impl_.get(), member_type::alloc_impl_.get())
  {}

  template <typename Tag>
  basic_value(Tag tag, allocator_type& alloc, typename details::enable_if< details::is_tag<Tag> >::type* = 0)
    : member_type(new native_value_type(Tag::native_value))
    , base_type(member_type::value_impl_.get(), &alloc)
  {}


  /*
   *        Value based constructors
   */
  template <typename T>
  basic_value(const T& value, typename details::disable_if< details::is_value_ref<T> >::type* = 0, typename details::disable_if< details::is_tag<T> >::type* = 0, typename details::disable_if< details::is_string<T> >::type* = 0, typename details::disable_if< details::is_cstr_ptr<T> >::type* = 0)
    : member_type(new native_value_type(value), new allocator_type())
    , base_type(member_type::value_impl_.get(), member_type::alloc_impl_.get())
  {}

  //Special handling for string types because to copy a string type we need to provide an allocator, but we don't have an allocator till we are part way through construction
  template <typename T>
  basic_value(const T& value, typename details::disable_if< details::is_value_ref<T> >::type* = 0, typename details::disable_if< details::is_tag<T> >::type* = 0, typename details::enable_if< details::is_string<T> >::type* = 0)
    : member_type(new native_value_type(null_tag::native_value), new allocator_type())
    , base_type(member_type::value_impl_.get(), member_type::alloc_impl_.get())
  {
    base_type::set(value);
  }

  //Special handling for string types because to copy a string type we need to provide an allocator, but we don't have an allocator till we are part way through construction
  template <typename T>
  basic_value(const T& value, typename details::disable_if< details::is_value_ref<T> >::type* = 0, typename details::disable_if< details::is_tag<T> >::type* = 0, typename details::enable_if<details::is_cstr_ptr<T> >::type* = 0)
    : member_type(new native_value_type(null_tag::native_value), new allocator_type())
    , base_type(member_type::value_impl_.get(), member_type::alloc_impl_.get())
  {
    base_type::set(value);
  }

  /*
   *        Value based constructors WITH an external allocator
   */
  template <typename T>
  basic_value(const T& value, allocator_type& alloc, typename details::disable_if< details::is_value_ref<T> >::type* = 0, typename details::disable_if< details::is_tag<T> >::type* = 0, typename details::disable_if< details::is_string<T> >::type* = 0, typename details::disable_if< details::is_cstr_ptr<T> >::type* = 0)
    : member_type(new native_value_type(value))
    , base_type(member_type::value_impl_.get(), &alloc)
  {}

  //Special handling to make sure we copy strings with the given allocator
  template <typename T>
  basic_value(const T& value, allocator_type& alloc, typename details::disable_if< details::is_value_ref<T> >::type* = 0, typename details::disable_if< details::is_tag<T> >::type* = 0, typename details::enable_if< details::is_string<T> >::type* = 0)
    : member_type(new native_value_type(value, alloc))
    , base_type(member_type::value_impl_.get(), &alloc)
  {}

  //Special handling to make sure we copy strings with the given allocator
  template <typename T>
  basic_value(const T& value, allocator_type& alloc, typename details::disable_if< details::is_value_ref<T> >::type* = 0, typename details::disable_if< details::is_tag<T> >::type* = 0, typename details::enable_if< details::is_cstr_ptr<T> >::type* = 0)
    : member_type(new native_value_type(value, alloc))
    , base_type(member_type::value_impl_.get(), &alloc)
  {}


  basic_value(const basic_value& other)
    : member_type()
    , base_type(other)
  {
    if (other.is_root_value())
      throw std::runtime_error("can not copy root value");
  }

  basic_value& operator=(const basic_value& other)
  {
    if (other.is_root_value())
      throw std::runtime_error("can not copy root value");
    basic_value(other).swap(*this);
    return *this;
  }

  template <typename OtherTraits>
  basic_value(const basic_value_ref<OtherTraits>& other)
    : member_type()
    , base_type(other)
  {
    if (base_type::is_null())
      traits::set(*this, default_tag());
  }

  template <typename OtherTraits>
  basic_value& operator=(const basic_value_ref<OtherTraits>& other)
  {
    basic_value(other).swap(*this);
    return *this;
  }


  template <typename T>
  basic_value& operator=(const T& value)
  {
    base_type::set(value);
    return *this;
  }

  void clear()
  {
    base_type::set(default_tag());
  }

  void swap(basic_value& other) throw()
  {
    base_type::swap(other);
    member_type::value_impl_.swap(other.value_impl_);
    member_type::alloc_impl_.swap(other.alloc_impl_);
  }

private:
  bool is_root_value() const
  {
    return member_type::value_impl_.get() != 0
        || member_type::alloc_impl_.get() != 0;
  }
};

template <typename Traits>
class basic_object : public basic_value<Traits, object_tag>
{
public:
  typedef basic_value<Traits, object_tag>               base_type;

  typedef typename base_type::encoding_type             encoding_type;
  typedef typename base_type::native_type               native_type;
  typedef typename base_type::native_document_type      native_document_type;
  typedef typename base_type::native_value_type         native_value_type;
  typedef typename base_type::native_allocator_type     native_allocator_type;

  typedef typename base_type::value_ref_type            value_ref_type;
  typedef typename base_type::const_value_ref_type      const_value_ref_type;
  typedef typename base_type::char_type                 char_type;
  typedef typename base_type::string_type               string_type;
  typedef typename base_type::string_ref_type           string_ref_type;
  typedef typename base_type::allocator_type            allocator_type;

  typedef typename base_type::member_iterator           iterator;
  typedef typename base_type::const_member_iterator     const_iterator;

public:
  basic_object()
    : base_type()
  {}

  basic_object(allocator_type& alloc)
    : base_type(alloc)
  {}

  basic_object(const basic_object& other)
    : base_type(other)
  {}

  template <typename OtherTraits>
  basic_object(const basic_value_ref<OtherTraits>& other)
    : base_type(other)
  {}

  iterator begin()                { return base_type::member_begin(); }
  iterator end()                  { return base_type::member_end(); }
  const_iterator begin() const    { return base_type::member_begin(); }
  const_iterator end() const      { return base_type::member_end(); }
  const_iterator cbegin() const   { return base_type::member_begin(); }
  const_iterator cend() const     { return base_type::member_end(); }
};

template <typename Traits>
class basic_array : public basic_value<Traits, array_tag>
{
public:
  typedef basic_value<Traits, array_tag>                base_type;

  typedef typename base_type::encoding_type             encoding_type;
  typedef typename base_type::native_type               native_type;
  typedef typename base_type::native_document_type      native_document_type;
  typedef typename base_type::native_value_type         native_value_type;
  typedef typename base_type::native_allocator_type     native_allocator_type;

  typedef typename base_type::value_ref_type            value_ref_type;
  typedef typename base_type::const_value_ref_type      const_value_ref_type;
  typedef typename base_type::char_type                 char_type;
  typedef typename base_type::string_type               string_type;
  typedef typename base_type::string_ref_type           string_ref_type;
  typedef typename base_type::allocator_type            allocator_type;

  typedef typename base_type::value_iterator            iterator;
  typedef typename base_type::const_value_iterator      const_iterator;

public:
  basic_array()
    : base_type()
  {}

  basic_array(allocator_type& alloc)
    : base_type(alloc)
  {}

  basic_array(const basic_array& other)
    : base_type(other)
  {}

  template <typename OtherTraits>
  basic_array(const basic_value_ref<OtherTraits>& other)
    : base_type(other)
  {}

  iterator begin()                { return base_type::value_begin(); }
  iterator end()                  { return base_type::value_end(); }
  const_iterator begin() const    { return base_type::value_begin(); }
  const_iterator end() const      { return base_type::value_end(); }
  const_iterator cbegin() const   { return base_type::value_begin(); }
  const_iterator cend() const     { return base_type::value_end(); }
};


template <typename Traits>
struct basic_document_base
{
  typedef basic_value_ref<Traits>                   base_type;
  typedef typename base_type::native_document_type  native_document_type;

  details::scoped_ptr<native_document_type> document_impl_;

  explicit basic_document_base(native_document_type* document = 0)
    : document_impl_(document)
  {}
};

template <typename Traits>
class basic_document
  : private basic_document_base<Traits>
  , public basic_value_ref<Traits>
{
public:
  typedef basic_document_base<Traits>                   member_type;
  typedef basic_value_ref<Traits>                       base_type;

  typedef typename base_type::encoding_type             encoding_type;
  typedef typename base_type::native_type               native_type;
  typedef typename base_type::native_document_type      native_document_type;
  typedef typename base_type::native_value_type         native_value_type;
  typedef typename base_type::native_allocator_type     native_allocator_type;

  typedef typename base_type::value_ref_type            value_ref_type;
  typedef typename base_type::const_value_ref_type      const_value_ref_type;
  typedef typename base_type::char_type                 char_type;
  typedef typename base_type::string_type               string_type;
  typedef typename base_type::string_ref_type           string_ref_type;
  typedef typename base_type::allocator_type            allocator_type;

private:
  basic_document(const basic_document&);
  basic_document& operator=(const basic_document&);

public:
  basic_document()
    : member_type(new native_document_type())
    , base_type(member_type::document_impl_.get(), &(member_type::document_impl_->GetAllocator()))
  {}

  void swap(basic_document& other)
  {
    base_type::swap(other);
    member_type::document_impl_.swap(other.document_impl_);
  }

  void parse(const string_ref_type& str)
  {
    parse<0>(str);
  }

  template <unsigned ParseFlags>
  void parse(const string_ref_type& str)
  {
    member_type::document_impl_->template Parse<ParseFlags>(str.data());
    if (member_type::document_impl_->HasParseError())
      throw parse_error(member_type::document_impl_->GetParseError());
  }
};


template <typename Traits>
void swap(basic_value_ref<Traits>& a, basic_value_ref<Traits>& b)
{
  a.swap(b);
}

template <typename Traits>
void swap(basic_value<Traits>& a, basic_value<Traits>& b)
{
  a.swap(b);
}

template <typename Traits>
void swap(basic_object<Traits>& a, basic_object<Traits>& b)
{
  a.swap(b);
}

template <typename Traits>
void swap(basic_array<Traits>& a, basic_array<Traits>& b)
{
  a.swap(b);
}

template <typename Traits>
void swap(basic_document<Traits>& a, basic_document<Traits>& b)
{
  a.swap(b);
}


template <typename Traits>
typename basic_value_ref<Traits>::string_type str(const basic_value_ref<Traits>& value)
{
  return value.str();
}

template <typename Char, typename Traits>
std::basic_ostream<Char>& operator<<(std::basic_ostream<Char>& os, const basic_value_ref<Traits>& value)
{
  os << str(value);
  return os;
}


typedef rapidjson::UTF8<> default_encoding;

template <typename Encoding = default_encoding>
struct types
{
  typedef details::value_ref_traits<Encoding>         traits;
  typedef details::const_value_ref_traits<Encoding>   const_traits;

  typedef basic_value_ref<traits>                     value_ref;
  typedef const basic_value_ref<const_traits>         const_value_ref;
  typedef basic_value<traits>                         value;
  typedef const basic_value<const_traits>             const_value;
  typedef basic_object<traits>                        object;
  typedef const basic_object<const_traits>            const_object;
  typedef basic_array<traits>                         array;
  typedef const basic_array<const_traits>             const_array;
  typedef basic_document<traits>                      document;
  typedef const basic_document<const_traits>          const_document;
  typedef typename document::allocator_type           allocator;
};

typedef types<>::value_ref        value_ref;
typedef types<>::const_value_ref  const_value_ref;
typedef types<>::value            value;
typedef types<>::const_value      const_value;
typedef types<>::object           object;
typedef types<>::const_object     const_object;
typedef types<>::array            array;
typedef types<>::const_array      const_array;
typedef types<>::document         document;
typedef types<>::const_document   const_document;
typedef types<>::allocator        allocator;

}

#endif
