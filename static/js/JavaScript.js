
const inde_vars_PE = document.getElementById("PE");

function handleClick(myRadio) {
    if(myRadio.value != null){
        inde_vars_PE.value.disabled = true;
    }    
}
//function handleClick(myRadio) {
//    if(myRadio.value != null){
//        var selectedValue = myRadio.value;
//
//        Object.values(inde_vars).forEach((value) => {
//            var text = value.nextElementSibling.textContent;
//            text = text.replace("(Recommand)", "");
//
////political_efficacy , informational_use, entertainment_use,SIU
//            if(selectedValue == value.value){
//                value.disabled = true;
//                value.checked = false;
//                value.nextElementSibling.style.color = "#c0c0c0"
//            }else{
//                value.disabled = false;
//                text += "(Recommand)";
//                value.nextElementSibling.style.color = "#002366";
//            }
//            value.nextElementSibling.textContent = text;
//          })
//    }
//}