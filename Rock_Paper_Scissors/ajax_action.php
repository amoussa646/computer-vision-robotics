<?php
include_once '../vars.php';
	
$msg=$_POST["message"];

if ($msg=="paper"){
	exec("sudo python3 /home/pi/Desktop/i2c.py 10");

	
}
elseif ($msg=="scissors"){
	exec("sudo python3 /home/pi/Desktop/i2c.py 8");
}
elseif ($msg=="rock"){
	exec("sudo python3 /home/pi/Desktop/i2c.py 9");
}


?>
