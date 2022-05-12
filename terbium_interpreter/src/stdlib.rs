use super::{TerbiumExceptionType, TerbiumObject};

pub fn terbium_default_repr(args: Vec<TerbiumObject>) -> Result<TerbiumObject, TerbiumExceptionType> {
    
}

pub fn terbium_print(args: Vec<TerbiumObject>) -> Result<TerbiumObject, TerbiumExceptionType>  {
    if args.len() < 1 {
        return Err(TerbiumExceptionType::SignatureError("print takes at least one argument".to_string()));
    }

    print!("{}", args[0].to_string()); // TODO of course this shouldn't just use Display ({})
    Ok(TerbiumObject::NULL)
}

pub fn terbium_println(args: Vec<TerbiumObject>) -> Result<TerbiumObject, TerbiumExceptionType>  {
    if args.len() < 1 {
        return Err(TerbiumExceptionType::SignatureError("println takes at least one argument".to_string()));
    }

    println!("{}", args[0].to_string());
    Ok(TerbiumObject::NULL)
}