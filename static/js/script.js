
const inde_vars_pe = document.getElementById('PE');// independent variable for form 1
const inde_vars = document.getElementsByName('idv');// independent variables for form 2

var inde_vars_name = [  "Online Expressive Political Participation (OEP)",
                        "Political Efficency",
                        "Informational Use",
                        "Socail Interactional Use(SIU)",
                        "Entertainment Purpose",
                        "Three Usage",
                        "Gender",
                        "Age",
                        "Education",
                        "Income",
                        "Location"];
var model_1 = [2,3,4];
var model_2 = [1];
var geo_data = [6,7,8,9,10];
//function for form 1
function handleClickPe(radio){
    if(radio.value == "political_efficacy"){
        inde_vars_pe.disabled = true;  
        inde_vars_pe.nextElementSibling.style.color = "#c0c0c0";
    }else{
        inde_vars_pe.disabled = false;
        inde_vars_pe.nextElementSibling.style.color = "#000000"
    }
    

}


//function for form 2
function handleClick(radio) {
    clear();
    if(radio.value == "OEP")
    {
        //disable OEP
        inde_vars[0].disabled = true;
        inde_vars[0].nextElementSibling.style.color = "#c0c0c0";
        //disable Political Efficacy
        //inde_vars[1].disabled = true;
        //inde_vars[1].nextElementSibling.style.color = "#c0c0c0";

        //Recommanded text for model 1
        model_1.forEach((value)=>{            
            inde_vars[value].nextElementSibling.textContent +=" (model-1)";
        })
        //Recommanded text for model 2
        model_2.forEach((value)=>{            
            inde_vars[value].nextElementSibling.textContent +=" (model-4)";
        })

        //Recommanded text for geographic data
        geo_data.forEach((value)=>{
            inde_vars[value].nextElementSibling.textContent += " (Recommand)";
        })
    }
    else if(radio.value == "OFEP")
    {
        model_1.forEach((value)=>{
            inde_vars[value].nextElementSibling.textContent +=" (model-2)";
        })
        //Recommanded text for model 2
        model_2.forEach((value)=>{
            inde_vars[value].nextElementSibling.textContent +=" (model-5)";
        })

        //Recommanded text for geographic data
        geo_data.forEach((value)=>{
            inde_vars[value].nextElementSibling.textContent += " (Recommand)";
        })
    }
    else 
    {
        //disable OEP
        inde_vars[0].disabled = true;
        inde_vars[0].nextElementSibling.style.color = "#c0c0c0";
        //disable Political Efficacy
        inde_vars[1].disabled = true;
        inde_vars[1].nextElementSibling.style.color = "#c0c0c0";

         model_1.forEach((value)=>{
            inde_vars[value].nextElementSibling.textContent +=" (model-3)";
        })
        //Recommanded text for geographic data
        geo_data.forEach((value)=>{
            inde_vars[value].nextElementSibling.textContent += " (Recommand)";
        })
    }
   
}

function clear(){
    var i = 0;
    Object.values(inde_vars).forEach((value) => {
        value.disabled = false;
        value.nextElementSibling.style.color = "#000000";        
        value.nextElementSibling.textContent = inde_vars_name[i];
        i++;
    })
}


// if(myRadio.value != null){
//     var selectedValue = myRadio.value;
    
//     Object.values(inde_vars).forEach((value) => {
//         var text = value.nextElementSibling.textContent;
//         text = text.replace("(Recommand)", "");

//         if(selectedValue == value.value){
//             value.disabled = true;                
//             value.checked = false;
//             value.nextElementSibling.style.color = "#c0c0c0"
//         }else{
//             value.disabled = false;                
//             text += "(Recommand)"; 
//             value.nextElementSibling.style.color = "#002366";               
//         }
//         value.nextElementSibling.textContent = text;            
//       })        
// }    