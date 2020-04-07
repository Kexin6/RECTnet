import React from 'react';
import Link from '@material-ui/core/Link';
import { makeStyles } from '@material-ui/core/styles';
import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableHead from '@material-ui/core/TableHead';
import TableRow from '@material-ui/core/TableRow';
import Title from './Title';

// Generate Order Data
function createData(id, date, name, location, paymentMethod, amount) {
  return { id, date, name, location, paymentMethod, amount };
}

const rows = [
  createData(0, '26 Mar, 2020', 'Cassie Li', 'Toronto, ON', 'VISA ⠀•••• 3719', 312.44),
  createData(1, '16 Mar, 2020', 'Cassie Li', 'Toronto, ON', 'VISA ⠀•••• 3719', 866.99),
  createData(2, '16 Mar, 2020', 'Cassie Li', 'Toronto, ON', 'VISA ⠀•••• 3719', 100.81),
  createData(3, '16 Mar, 2020', 'Cassie Li', 'Toronto, ON', 'VISA ⠀•••• 3719', 654.39),
  createData(4, '15 Mar, 2020', 'Cassie Li', 'Toronto, ON', 'VISA ⠀•••• 3719', 212.79),
];

function preventDefault(event) {
  event.preventDefault();
}

const useStyles = makeStyles((theme) => ({
  seeMore: {
    marginTop: theme.spacing(3),
  },
}));

export default function Orders() {
  const classes = useStyles();
  return (
    <React.Fragment>
      <Title>Recent Usage</Title>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Date</TableCell>
            <TableCell>Name</TableCell>
            <TableCell>Location</TableCell>
            <TableCell>Payment Method</TableCell>
            <TableCell align="right"> Cost </TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {rows.map((row) => (
            <TableRow key={row.id}>
              <TableCell>{row.date}</TableCell>
              <TableCell>{row.name}</TableCell>
              <TableCell>{row.location}</TableCell>
              <TableCell>{row.paymentMethod}</TableCell>
              <TableCell align="right">{row.amount}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <div className={classes.seeMore}>
 
      </div>
    </React.Fragment>
  );
}
