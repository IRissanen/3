package web

// THIS FILE IS AUTO GENERATED FROM webgui.js
// EDITING IS FUTILE

const templText = `
<!DOCTYPE html>
<html>
<head>
	<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
	<title>mx3</title>
	<style media="all" type="text/css">
		body { margin: 20px; font-family: Ubuntu, Arial, sans-serif; font-size: 14px; color: #444444}
		h1   { font-size: 22px; font-weight: normal; color: black}
		img  { margin: 15px; }
		table{ border:"10"; }
		hr   { border-style: none; border-top: 1px solid #CCCCCC; }
		a    { color: #375EAB; text-decoration: none; }
		a#hide{ color: black; font-size:17px; text-decoration: none; cursor: pointer; font-weight: normal; }
		input#text{ border:solid; border-color:#BBBBBB; border-width:1px; padding-left:4px;}
		div  { margin-left: 20px; margin-top: 10px; margin-bottom: 20px; }
		div#header{ color:gray; font-size:16px; }
		div#footer{ color:gray; font-size:14px; }
		.err { color: red; font-weight: bold; } 
	</style>
</head>


<body>

<h1> {{.Version}} </h1> 
<span id=err_0 class=err></span>
<hr/>

<script>
	function http(method, url){
    	var xmlHttp = new XMLHttpRequest();
    	xmlHttp.open(method, url, false);
    	xmlHttp.send(null);
    	return xmlHttp.responseText;
    }
	function httpGet(url){
		return http("GET", url);
	}
	function httpPost(url){
		return http("POST", url);
	}
</script>

<script>
	var running = false;
	function updateRunning(){
		try{
			running = (httpGet("/running/") === "true");
			document.getElementById("err_0").innerHTML = "";
		}catch(err){
			running = false	;
			document.getElementById("err_0").innerHTML = "Disconnected from simulation";
		}
		if(running){
			document.getElementById("running").innerHTML = "<b>Running</b>";
		}else{
			document.getElementById("running").innerHTML = "<b>Paused</b>";
		}
	}
	setInterval(updateRunning, 200);
</script>

<script>
	function rpc(command){
		httpPost("/script/" + command);
	}

	// adds a button and as many text boxes as args (default arguments).
	// on click the rpc command is called with the text boxes input as arguments.
	function rpcCmd(command, args, infix, err){
		var par = document.scripts[document.scripts.length - 1].parentNode;

		var submit = function(){
			var arg = "(";
			for (var i=0; i < boxes.length; i++){
				arg += boxes[i].value;
				if (i != boxes.length - 1) { arg += ","; }
			}
			arg += ")";
			var errArea = document.getElementById(err);
			errArea.innerHTML = "";
			try{
				rpc(command + infix + arg);
			}catch(e){
				errArea.innerHTML = e;
			}
		};

		var button;
		if (args.length == 0){
			button = document.createElement("input");
			button.type = 'button';
			button.value = command;
		} else {
			button = document.createElement("td");
			button.innerHTML = command;
			button.style.cursor = 'pointer';
		}
		button.onclick = submit;
		par.appendChild(button);

		var boxes = [];
		for (var i=0; i < args.length; i++){
			var textbox = document.createElement("input");
			textbox.type = "text";
			textbox.value = args[i];
			textbox.id = "text";
			textbox.size = 10;
			textbox.onkeydown=function(){
				if (event.keyCode == 13) { submit(); } // retrun key
			};
			par.appendChild(textbox);
			boxes.push(textbox);
		}
	}

	function rpcCall(command, args, err){
		rpcCmd(command, args, "", err)
	}

	function rpcSet(command, args, err){
		rpcCmd(command, args, "=", err)
	}

	// add a button that executes an rpc command
	function rpcButton(command, err){
		rpcCall(command, [], err);
	}
 
</script>



<script>
	function toggle(id) {
       var el = document.getElementById(id);
       if(el.style.display != 'none'){
          el.style.display = 'none';
		}
       else
          el.style.display = 'block';
    }
</script>


<a id=hide onclick="toggle('div_1');"> Initialize <br/></a> 
<div id=div_1>
<table>
	<tr><script> rpcCall("setGridSize", [128, 32, 1], "err_1");        </script> <td>(cells)</td> </tr>
	<tr><script> rpcCall("setCellSize", [3e-9, 3e-9, 5e-9], "err_1");  </script> <td>(m)    </td> </tr>
	<tr><script> rpcSet("m", ["uniform(1, 1, 0)"], "err_1");           </script>  </tr>
	<tr><script> rpcSet("geom", ["rect()"], "err_1");                  </script>  </tr>
</table>
<span id=err_1 class=err></span>
</div>


<a id=hide onclick="toggle('div_solver');"> Run <br/></a> 
<div id=div_solver>
<table><tr><td>  


<table>
	<tr><script> rpcCall("Relax", [1e-5], "err_2"); </script> <td>maxtorque</td> </tr>
	<tr><script> rpcCall("Run", [1e-9], "err_2");   </script> <td>seconds</td> </tr>
	<tr><script> rpcCall("Steps", [1000], "err_2"); </script>            </tr>
	<tr><script> rpcButton("Pause", "err_2");       </script>            </tr>
</table>

</td><td>  
 &nbsp; &nbsp; &nbsp;
</td><td>  
	<span id="running"><b>Paused</b></span> 
	<span id="dash"> </span>
</td></tr></table>
<span id=err_2 class=err></span>

<script>
	function updateDash(){
		try{
			document.getElementById("dash").innerHTML = httpGet("/dash/");
		}catch(err){}
	}
	function updateDashIfRunning(){
		if(running){
			updateDash();
		}
	}
	updateDash();
	setInterval(updateDashIfRunning, 200);
</script>

</div>


<a id=hide onclick="toggle('div_disp');"> Output <br/></a> 
<div id=div_disp>
<script>
	var renderQuant = "m"; 
	var renderComp = ""; 
	var img = new Image();
	img.src = "/render/" + renderQuant;

	function updateImg(){
		var renderQuantComp = renderQuant;
		if(renderComp != ""){
			renderQuantComp += "/" + renderComp;
		}	
		img.src = "/render/" + renderQuantComp + "?" + new Date(); // date = cache breaker
		document.getElementById("display").src = img.src;
	}

	function updateImgAsync(){
		if(running && img.complete){
			updateImg();
		}
	}

	setInterval(updateImgAsync, 500);

	function renderSelect() {
		var list=document.getElementById("renderList");
		renderQuant=list.options[list.selectedIndex].text;
		list=document.getElementById("renderComp");
		renderComp=list.options[list.selectedIndex].text;
		updateImg();
	}
</script>

Display: <select id="renderList" onchange="renderSelect()">
		{{range $k, $v := .Quants}}
  			<option>{{$k}}</option>
		{{end}}
	</select>
	<select id="renderComp" onchange="renderSelect()">
  		<option></option>
  		<option>x</option>
  		<option>y</option>
  		<option>z</option>
	</select> <br/>

<script> document.getElementById("renderList").value = renderQuant; </script>

<img id="display" src="/render/m" alt="display"/>

</div>

<a id=hide onclick="toggle('div_out');"> output <br/></a> 
<div id=div_out>


</div>


<div id="footer">
<hr/>
<br/>
<center>
{{.Uname}}<br/>
Copyright 2012-2013 <a href="mailto:a.vansteenkiste@gmail.com">Arne Vansteenkiste</a>, <a href="http://dynamat.ugent.be">Dynamat group</a>, Ghent University, Belgium.<br/>
You are free to modify and distribute this software under the terms of the <a href="http://www.gnu.org/licenses/gpl-3.0.html">GPLv3 license</a>.
</center>
</div>

</body>
</html>
`
